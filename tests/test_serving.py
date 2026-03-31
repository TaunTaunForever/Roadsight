import asyncio

from src.serving import app as serving_app


def test_predict_image_endpoint_uses_trained_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        serving_app,
        "load_yolo_model",
        lambda weights: object(),
    )
    monkeypatch.setattr(
        serving_app,
        "predict_with_model",
        lambda inference, source, run_name, save=False: {
            "results": [
                type(
                    "Result",
                    (),
                    {
                        "names": {0: "car"},
                        "boxes": type(
                            "Boxes",
                            (),
                            {
                                "xyxy": [[1.0, 2.0, 3.0, 4.0]],
                                "conf": [0.9],
                                "cls": [0.0],
                            },
                        )(),
                    },
                )()
            ]
        },
    )

    app = serving_app.create_app()
    predict_route = next(route for route in app.routes if getattr(route, "path", None) == "/predict/image")

    class FakeRequest:
        headers = {"x-filename": "frame.jpg", "content-type": "application/octet-stream"}

        async def body(self):
            return b"fake-image-bytes"

    payload = asyncio.run(predict_route.endpoint(FakeRequest()))
    assert payload["model_weights"].endswith("weights/best.pt")
    assert payload["source_filename"] == "frame.jpg"
    assert payload["detections"] == [
        {
            "class_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "box_xyxy": [1.0, 2.0, 3.0, 4.0],
        }
    ]
