from src.reasoning.session_memory import PersonaSessionMemory


def test_session_memory_should_keep_recent_three_turns() -> None:
    memory = PersonaSessionMemory(ttl_sec=60, max_people=2)
    memory.update("p1", "天气", "neutral", "你好")
    memory.update("p1", "出行", "happy", "今天天气不错")
    memory.update("p1", "音乐", "happy", "来点音乐")
    memory.update("p1", "电影", "neutral", "推荐电影")
    summary = memory.summarize("p1")
    assert "电影" in summary
