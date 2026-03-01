"""已重构: test_v2_agents | V2 | 2026-03-01"""

import unittest

from src.agents.followup_policy_agent import FollowupPolicyAgent
from src.agents.intent_agent import IntentAgent
from src.agents.reply_agent import ReplyAgent
from src.agents.reply_style_guard import ReplyStyleGuard
from src.core.chat_orchestrator import ChatOrchestrator
from src.domain.chat_models import ConversationContext, FollowupDecision, IntentResult, IntentType


class _StubKnowledgeService:
    def __init__(self, detail_map=None):
        self.detail_map = detail_map or {}

    def find_answer_detail(self, user_message: str, threshold: float = 0.65):
        del threshold
        key = str(user_message or "").strip()
        return self.detail_map.get(key, {"matched": False, "score": 0.0, "mode": "none", "item_id": ""})

    class _Repo:
        def __init__(self, items):
            self._items = items

        def get_all(self):
            return list(self._items)

    @property
    def repository(self):
        class _Item:
            def __init__(self, item_id, question, answer):
                self.id = item_id
                self.question = question
                self.answer = answer
                self.answers = [answer]

        return self._Repo(
            [
                _Item("c1", "冬天可以戴假发吗", "冬天可以佩戴，注意防静电护理。"),
                _Item("c2", "价格多少", "一般在3000-6000区间，需按头围脸型评估。"),
            ]
        )


class _StubLLMService:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompt = ""

    def set_system_prompt(self, prompt: str):
        self.prompt = prompt

    def generate_reply_sync(self, user_message: str, conversation_history=None):
        del user_message, conversation_history
        if not self.replies:
            return True, "冬天可以戴假发，注意头皮保湿和防静电。"
        return True, self.replies.pop(0)


class V2AgentTestCase(unittest.TestCase):
    def test_intent_agent_classifies_pre_sales(self):
        agent = IntentAgent()
        result = agent.analyze("请问价格多少，门店在哪", [])
        self.assertEqual(result.intent, IntentType.PRE_SALES)
        self.assertGreater(result.confidence, 0.4)

    def test_followup_limit_max_two_rounds(self):
        agent = FollowupPolicyAgent()
        intent = IntentResult(intent=IntentType.PRE_SALES, confidence=0.8, signals=[])
        first = agent.decide(intent, {"followup_count": 0}, "我想去门店看看")
        second = agent.decide(intent, {"followup_count": 1}, "我想去门店看看")
        third = agent.decide(intent, {"followup_count": 2}, "我想去门店看看")
        self.assertTrue(first.need_followup)
        self.assertTrue(second.need_followup)
        self.assertFalse(third.need_followup)

    def test_style_guard_sanitize_accepts_kb_sentence(self):
        guard = ReplyStyleGuard()
        text, sanitized = guard.sanitize("冬天可以戴，注意防静电😊。")
        report = guard.validate(text, mode="light")
        self.assertTrue(report.passed)
        self.assertTrue(sanitized)

    def test_price_first_hit_uses_kb_direct(self):
        ks = _StubKnowledgeService(
            {
                "你们的价格都是多少？": {
                    "matched": True,
                    "answer": "一般在3000-6000区间，需按头围脸型评估。",
                    "answers": [
                        "一般在3000-6000区间，需按头围脸型评估。",
                        "价格通常在3000到6000之间，具体看定制需求。",
                    ],
                    "score": 0.92,
                    "mode": "contains",
                    "item_id": "price_1",
                }
            }
        )
        agent = ReplyAgent(knowledge_service=ks, llm_service=_StubLLMService([]), kb_high_confidence=0.65)
        context = ConversationContext(
            session_id="s",
            user_name="u",
            latest_user_text="你们的价格都是多少？",
            history=[],
            state={"v2_kb_seen_count_by_item": {}, "v2_recent_kb_answer_hashes": []},
        )
        candidate = agent.generate_candidate(
            context=context,
            intent_result=IntentResult(intent=IntentType.PRE_SALES, confidence=0.8, signals=[]),
            followup_decision=FollowupDecision(need_followup=False, followup_text="", reason="direct_answer"),
        )
        self.assertEqual(candidate.source, "kb_direct")
        self.assertEqual(candidate.kb_meta.get("answer_index"), 0)

    def test_price_second_hit_random_non_repeat(self):
        ks = _StubKnowledgeService(
            {
                "你们的价格都是多少？": {
                    "matched": True,
                    "answer": "一般在3000-6000区间，需按头围脸型评估。",
                    "answers": [
                        "一般在3000-6000区间，需按头围脸型评估。",
                        "价格通常在3000到6000之间，具体看定制需求。",
                    ],
                    "score": 0.92,
                    "mode": "contains",
                    "item_id": "price_1",
                }
            }
        )
        agent = ReplyAgent(knowledge_service=ks, llm_service=_StubLLMService([]), kb_high_confidence=0.65)
        context = ConversationContext(
            session_id="s",
            user_name="u",
            latest_user_text="你们的价格都是多少？",
            history=[],
            state={
                "v2_kb_seen_count_by_item": {"price_1": 1},
                "v2_recent_kb_answer_hashes": ["一般在3000-6000区间，需按头围脸型评估。"],
            },
        )
        candidate = agent.generate_candidate(
            context=context,
            intent_result=IntentResult(intent=IntentType.PRE_SALES, confidence=0.8, signals=[]),
            followup_decision=FollowupDecision(need_followup=False, followup_text="", reason="direct_answer"),
        )
        self.assertEqual(candidate.source, "kb_direct")
        self.assertNotEqual(candidate.text, "一般在3000-6000区间，需按头围脸型评估。")

    def test_low_confidence_hit_goes_llm(self):
        ks = _StubKnowledgeService(
            {
                "冬天可以戴假发吗？": {
                    "matched": True,
                    "answer": "冬天可以佩戴，注意防静电护理。",
                    "answers": ["冬天可以佩戴，注意防静电护理。"],
                    "score": 0.45,
                    "mode": "char_overlap",
                    "item_id": "wearing_1",
                }
            }
        )
        llm = _StubLLMService(["冬天可以戴假发，重点做好保湿和防静电。"])
        agent = ReplyAgent(knowledge_service=ks, llm_service=llm, kb_high_confidence=0.65)
        context = ConversationContext(
            session_id="s",
            user_name="u",
            latest_user_text="冬天可以戴假发吗？",
            history=[],
            state={},
        )
        candidate = agent.generate_candidate(
            context=context,
            intent_result=IntentResult(intent=IntentType.OTHER, confidence=0.5, signals=[]),
            followup_decision=FollowupDecision(need_followup=False, followup_text="", reason="direct_answer"),
        )
        self.assertEqual(candidate.source, "kb_low_conf_llm")
        self.assertAlmostEqual(float(candidate.kb_meta.get("score", 0.0)), 0.45)

    def test_orchestrator_no_hit_uses_llm_and_no_template_blank(self):
        intent_agent = IntentAgent()
        followup = FollowupPolicyAgent()
        reply_agent = ReplyAgent(
            knowledge_service=_StubKnowledgeService({}),
            llm_service=_StubLLMService(["冬天可以戴假发，注意头皮保湿和防静电。"]),
            kb_high_confidence=0.65,
        )
        guard = ReplyStyleGuard()
        orchestrator = ChatOrchestrator(intent_agent, followup, reply_agent, guard, max_regenerations=3)
        context = ConversationContext(
            session_id="s1",
            user_name="u1",
            latest_user_text="冬天可以戴假发吗？",
            history=[],
            state={"followup_count": 0},
        )
        result = orchestrator.process(context)
        self.assertIn("冬天", result.text)
        self.assertNotIn("说下重点", result.text)


if __name__ == "__main__":
    unittest.main()
