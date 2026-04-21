"""人脸分析真实链路测试。"""

from src.perception.face_analyzer import FaceAnalyzer


def test_face_analyzer_should_return_unknown_on_empty_frame() -> None:
    """空帧应返回 unknown。"""
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(b"")
    assert result.gender == "unknown"
    assert result.confidence == 0.0


def test_face_analyzer_should_handle_non_image_bytes() -> None:
    """非图像字节流应安全降级。"""
    analyzer = FaceAnalyzer()
    result = analyzer.analyze(b"not-an-image")
    assert result.gender == "unknown"
