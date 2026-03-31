from src.serving.app import create_app


def test_health_endpoint_returns_ok() -> None:
    app = create_app()
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    payload = health_route.endpoint()

    assert payload["status"] == "ok"
    assert payload["service"] == "roadsight"
    assert payload["model_weights"].endswith("weights/best.pt")
