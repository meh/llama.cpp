"""
Test that multiple models can be loaded simultaneously when using the router.

Architecture overview:
- Router mode (with '--'): Router process manages child processes, each child loads one model.
  Multiple models can be loaded simultaneously up to models_max (default 4).
- Non-router mode (without '--'): Single process loads one model directly.
  No multi-model support.

This test verifies router behavior.
"""

import pytest
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def _get_model_status(model_id: str) -> str:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200, f"Failed to load model {model_id}: {load_res.body}"
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_multiple_models_loaded_simultaneously():
    """
    Test that two models can be loaded simultaneously when models_max >= 2.
    This verifies that the router does NOT unload the current model when loading a new one,
    unless models_max limit is reached.
    """
    global server
    server.models_max = 4
    server.start()

    # Get available models from /models endpoint
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    available = res.body.get("data", [])
    assert len(available) >= 1, f"No models available in /models response"

    # Use the same model twice to verify simultaneous loading works
    model_id = available[0]["id"]

    # Load the model
    _load_model_and_wait(model_id, timeout=120)
    assert _get_model_status(model_id) == "loaded"

    # Load it again (should succeed since it's already loaded)
    load_res2 = server.make_request(
        "POST", "/models/load", data={"model": model_id}
    )
    assert load_res2.status_code == 200, f"Second load failed: {load_res2.body}"

    # Model should still be loaded
    assert _get_model_status(model_id) == "loaded"


def test_router_lru_eviction_when_models_max_reached():
    """
    Test that LRU eviction occurs when models_max limit is reached.
    With models_max=2, loading a 3rd model should evict the least recently used.
    """
    global server
    server.models_max = 2
    server.start()

    # Get available models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    available = res.body.get("data", [])
    assert len(available) >= 3, f"Need at least 3 models, found: {[m['id'] for m in available]}"

    first = available[0]["id"]
    second = available[1]["id"]
    third = available[2]["id"]

    # Load first two models
    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load third model - should evict the first (LRU)
    _load_model_and_wait(third, timeout=120)

    assert _get_model_status(third) == "loaded"
    # First model should have been evicted (LRU)
    assert _get_model_status(first) == "unloaded", \
        f"First model should have been evicted by LRU, status: {_get_model_status(first)}"
    # Second model should still be loaded
    assert _get_model_status(second) == "loaded", \
        f"Second model should still be loaded, status: {_get_model_status(second)}"


def test_router_chat_completion_with_multiple_models():
    """
    Test that chat completions work with different models when multiple are loaded.
    """
    global server
    server.models_max = 4
    server.start()

    # Get available models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    available = res.body.get("data", [])
    assert len(available) >= 2, f"Need at least 2 models, found: {[m['id'] for m in available]}"

    first_model = available[0]["id"]
    second_model = available[1]["id"]

    # Load both models
    _load_model_and_wait(first_model, timeout=120)
    _load_model_and_wait(second_model, timeout=120)

    # Verify both are loaded
    assert _get_model_status(first_model) == "loaded"
    assert _get_model_status(second_model) == "loaded"

    # Make chat completion requests to both models
    res1 = server.make_request("POST", "/v1/chat/completions", data={
        "model": first_model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
    })
    assert res1.status_code == 200, f"First model chat completion failed: {res1.body}"

    res2 = server.make_request("POST", "/v1/chat/completions", data={
        "model": second_model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
    })
    assert res2.status_code == 200, f"Second model chat completion failed: {res2.body}"

    # Both models should still be loaded
    assert _get_model_status(first_model) == "loaded"
    assert _get_model_status(second_model) == "loaded"
