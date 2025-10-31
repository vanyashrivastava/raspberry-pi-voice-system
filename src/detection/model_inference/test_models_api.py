import threading
import random
import numpy as np

from src.detection.model_inference.test_models import TestModels
from src.detection.model_inference.inference_engine import InferenceEngine
from src.detection.model_inference import load_model, run_inference, batch_inference, get_model_status

def setup_module(module):
    random.seed(42)
    np.random.seed(42)
    load_model(None)

def test_using_skeleton_smoke_methods():
    eng = InferenceEngine()
    eng.load_model(None)
    tm = TestModels(eng)
    assert tm.smoke_test_text_model() is True
    assert tm.smoke_test_email_model() is True

def test_run_inference_api_basic():
    res = run_inference("You have won a prize - claim now")
    assert isinstance(res, dict)
    assert "scam_probability" in res
    assert 0.0 <= res["scam_probability"] <= 100.0
    assert res["status"] == "ok"

def test_batch_inference_and_length():
    items = ["a", "b", "c"]
    out = batch_inference(items)
    assert isinstance(out, list)
    assert len(out) == len(items)
    for r in out:
        assert r["status"] == "ok"

def test_concurrent_calls_using_package_helpers():
    results = []
    def worker(i):
        results.append(run_inference(f"concurrent test {i}"))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(results) == 4
    for r in results:
        assert r["status"] == "ok"