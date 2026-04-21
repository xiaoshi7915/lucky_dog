"""API 集成测试。"""

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint_should_return_ok() -> None:
    """健康检查接口应返回运行状态。"""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "environment" in payload


def test_turn_endpoint_should_return_action_and_metrics() -> None:
    """单轮接口应返回动作执行与性能指标。"""
    response = client.post(
        "/v1/session/s1/turn",
        json={"asrText": "你好 靠近", "frameData": "frame-demo"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action_ok"] is True
    assert payload["dialogue"]["action_intent"] in {"approach", "nod_head", "wag_tail"}
    assert "metrics" in payload
