"""已重构: chat_orchestrator | V2 | 2026-03-01"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..agents.followup_policy_agent import FollowupPolicyAgent
from ..agents.intent_agent import IntentAgent
from ..agents.reply_agent import ReplyAgent
from ..agents.reply_style_guard import ReplyStyleGuard
from ..domain.chat_models import ConversationContext, FinalReply, ReplyCandidate
from ..services.knowledge_service import KnowledgeService
from ..services.llm_service import LLMService


class ChatOrchestrator:
    """V2 编排器：只负责流程，不承载策略细节。"""

    def __init__(
        self,
        intent_agent: IntentAgent,
        followup_policy_agent: FollowupPolicyAgent,
        reply_agent: ReplyAgent,
        reply_style_guard: ReplyStyleGuard,
        max_regenerations: int = 3,
    ):
        self.intent_agent = intent_agent
        self.followup_policy_agent = followup_policy_agent
        self.reply_agent = reply_agent
        self.reply_style_guard = reply_style_guard
        self.max_regenerations = max(1, int(max_regenerations))

    def process(self, input: ConversationContext) -> FinalReply:
        intent_result = self.intent_agent.analyze(input.latest_user_text, input.history)
        followup_decision = self.followup_policy_agent.decide(intent_result, input.state, input.latest_user_text)

        candidate = self.reply_agent.generate_candidate(input, intent_result, followup_decision)
        failed_reasons: List[str] = []
        sanitized_any = False

        for attempt in range(1, self.max_regenerations + 1):
            sanitized_text, sanitized = self.reply_style_guard.sanitize(candidate.text)
            sanitized_any = sanitized_any or sanitized
            mode = "light" if candidate.source in ("kb_direct", "followup", "address_route") else "standard"
            report = self.reply_style_guard.validate(
                sanitized_text,
                mode=mode,
                attempts=attempt,
                failed_reasons=failed_reasons,
            )
            if report.passed:
                return FinalReply(
                    text=sanitized_text,
                    intent=intent_result.intent,
                    source=candidate.source,
                    guard_report={
                        "passed": True,
                        "reason": "passed",
                        "attempts": attempt,
                        "failed_reasons": failed_reasons,
                        "sanitized": sanitized_any,
                        "generation_path": candidate.source,
                        "kb": candidate.kb_meta,
                        "intent_confidence": intent_result.confidence,
                        "intent_signals": intent_result.signals,
                        "followup": {
                            "need_followup": followup_decision.need_followup,
                            "reason": followup_decision.reason,
                        },
                    },
                )

            if not candidate.should_regenerate:
                return FinalReply(
                    text=sanitized_text,
                    intent=intent_result.intent,
                    source=candidate.source,
                    guard_report={
                        "passed": False,
                        "reason": report.reason,
                        "attempts": attempt,
                        "failed_reasons": failed_reasons + [report.reason],
                        "sanitized": sanitized_any,
                        "generation_path": candidate.source,
                        "kb": candidate.kb_meta,
                        "intent_confidence": intent_result.confidence,
                        "intent_signals": intent_result.signals,
                        "followup": {
                            "need_followup": followup_decision.need_followup,
                            "reason": followup_decision.reason,
                        },
                    },
                )

            failed_reasons.append(report.reason)
            candidate = self.reply_agent.regenerate_with_feedback(
                input,
                intent_result,
                followup_decision,
                failure_reason=report.reason,
                previous_candidate=candidate,
            )

        fallback_text = self.reply_agent.fallback_by_intent(intent_result.intent)
        fallback_text, fallback_sanitized = self.reply_style_guard.sanitize(fallback_text)
        fallback_report = self.reply_style_guard.validate(
            fallback_text,
            mode="standard",
            attempts=self.max_regenerations + 1,
            failed_reasons=failed_reasons,
        )
        final_text = fallback_text if fallback_report.passed else "这个问题需要结合具体情况进一步确认。"
        final_text, final_sanitized = self.reply_style_guard.sanitize(final_text)
        final_guard = self.reply_style_guard.validate(
            final_text,
            mode="standard",
            attempts=self.max_regenerations + 2,
            failed_reasons=failed_reasons,
        )
        return FinalReply(
            text=final_text,
            intent=intent_result.intent,
            source="template_fallback",
            guard_report={
                "passed": bool(final_guard.passed),
                "reason": final_guard.reason,
                "attempts": final_guard.attempts,
                "failed_reasons": failed_reasons,
                "sanitized": bool(sanitized_any or fallback_sanitized or final_sanitized),
                "generation_path": "template_fallback",
                "kb": {
                    "matched": False,
                    "score": 0.0,
                    "item_id": "",
                    "answer_index": -1,
                    "mode": "none",
                },
                "intent_confidence": intent_result.confidence,
                "intent_signals": intent_result.signals,
                "followup": {
                    "need_followup": followup_decision.need_followup,
                    "reason": followup_decision.reason,
                },
            },
        )


def build_v2_orchestrator(
    knowledge_service: KnowledgeService,
    llm_service: LLMService,
    max_regenerations: int = 3,
    kb_high_confidence: float = 0.65,
) -> ChatOrchestrator:
    return ChatOrchestrator(
        intent_agent=IntentAgent(),
        followup_policy_agent=FollowupPolicyAgent(),
        reply_agent=ReplyAgent(
            knowledge_service=knowledge_service,
            llm_service=llm_service,
            kb_high_confidence=kb_high_confidence,
        ),
        reply_style_guard=ReplyStyleGuard(),
        max_regenerations=max_regenerations,
    )
