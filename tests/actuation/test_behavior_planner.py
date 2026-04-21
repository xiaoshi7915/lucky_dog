from src.reasoning.action_policy import ActionPolicy


def test_action_policy_should_consider_negative_emotion() -> None:
    policy = ActionPolicy()
    decision = policy.decide(
        reply_text="我在这里",
        asr_text="",
        vision_summary="",
        active_person_id="p1",
        emotion_trend="negative",
        distance_delta=0.0,
    )
    assert decision.intent == "nod_head"
