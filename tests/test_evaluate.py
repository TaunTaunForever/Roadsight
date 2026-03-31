from src.models import evaluate as evaluate_module


def test_run_evaluation_calls_ultralytics(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeYOLO:
        def val(self, **kwargs):
            captured["kwargs"] = kwargs
            return "evaluated"

    monkeypatch.setattr(evaluate_module, "load_yolo_model", lambda weights: FakeYOLO())
    monkeypatch.setattr(evaluate_module, "resolve_training_device", lambda *args, **kwargs: "cpu")

    outcome = evaluate_module.run_evaluation(
        {
            "model_weights": "runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt",
            "data_config": "data/processed/bdd100k_yolo/dataset.yaml",
            "output_dir": "runs/inference",
            "eval_run_name": "evaluate",
            "confidence_threshold": 0.25,
            "save_json": False,
        }
    )

    assert captured["kwargs"] == {
        "data": "data/processed/bdd100k_yolo/dataset.yaml",
        "conf": 0.25,
        "project": "runs/inference",
        "name": "evaluate",
        "device": "cpu",
        "save_json": False,
        "split": "val",
    }
    assert outcome["results"] == "evaluated"
    assert outcome["save_dir"] == "runs/inference/evaluate"
