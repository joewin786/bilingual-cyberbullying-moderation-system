import requests
import json
import sys

API_URL = "http://localhost:8000"

def run_test_case(name, func):
    print(f"[*] Running: {name} ... ", end="")
    try:
        func()
        print("\033[92mPASS\033[0m")
        return True
    except AssertionError as e:
        print("\033[91mFAIL\033[0m")
        print(f"    Assertion Error: {e}")
        return False
    except Exception as e:
        print("\033[91mERROR\033[0m")
        print(f"    Exception: {e}")
        return False

def test_health_endpoint():
    resp = requests.get(f"{API_URL}/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "status" in data, "Missing 'status' in health response"
    assert "model_loaded" in data, "Missing 'model_loaded' in health response"
    assert data["model_loaded"] is True, "Model is not loaded on the API server!"

def test_predict_non_bully():
    payload = {"text": "Halo teman-teman, selamat pagi! Semangat ya hari ini.", "lang": "id"}
    resp = requests.post(f"{API_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["label"] == "non-bully", f"Expected non-bully, got {data['label']}"
    assert data["label_id"] == 0, f"Expected 0, got {data['label_id']}"
    assert data["action_tier"] == "ignore", f"Expected ignore, got {data['action_tier']}"

def test_predict_bully():
    payload = {"text": "dasar kamu bodoh bangsat anjing sekali", "lang": "id"}
    resp = requests.post(f"{API_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["label"] == "bully", f"Expected bully, got {data['label']}"
    assert data["label_id"] == 1, f"Expected 1, got {data['label_id']}"
    assert data["action_tier"] in ["flag", "action"], f"Expected flag or action, got {data['action_tier']}"

def test_predict_validation_empty():
    # Test sending empty string
    payload = {"text": "    ", "lang": "id"}
    resp = requests.post(f"{API_URL}/predict", json=payload)
    # Fastapi returns 422 for pydantic validation errors
    assert resp.status_code == 422, f"Expected 422 for validation error, got {resp.status_code}"

def test_predict_batch_valid():
    payload = {
        "texts": [
            "semangat ya semuanya!",
            "lu jelek banget anjing goblog"
        ],
        "lang": "id"
    }
    resp = requests.post(f"{API_URL}/predict/batch", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert len(data) == 2, f"Expected 2 predictions, got {len(data)}"
    assert data[0]["label"] == "non-bully"
    assert data[1]["label"] == "bully"

def test_predict_batch_exceed_limit():
    # Max batch size is 32. Let's send 33 items.
    payload = {
        "texts": ["test"] * 33,
        "lang": "id"
    }
    resp = requests.post(f"{API_URL}/predict/batch", json=payload)
    # Expected validation error
    assert resp.status_code == 422, f"Expected 422 for batch size limit error, got {resp.status_code}"

def main():
    print("=" * 60)
    print("CYBERBULLYING DETECTION API — BLACKBOX FUNCTIONAL TESTS")
    print("=" * 60)
    
    test_cases = [
        ("API Health check", test_health_endpoint),
        ("Single prediction (Non-Bully)", test_predict_non_bully),
        ("Single prediction (Bully)", test_predict_bully),
        ("Validation: Empty text", test_predict_validation_empty),
        ("Batch prediction (Valid)", test_predict_batch_valid),
        ("Validation: Batch size limit (>32)", test_predict_batch_exceed_limit),
    ]
    
    passed_count = 0
    for name, func in test_cases:
        if run_test_case(name, func):
            passed_count += 1
            
    print("=" * 60)
    print(f"Result: {passed_count}/{len(test_cases)} tests passed.")
    print("=" * 60)
    
    if passed_count < len(test_cases):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
