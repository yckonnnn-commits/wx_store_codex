"""已重构: reply_agent | V2 | 2026-03-01"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List

from ..domain.chat_models import ConversationContext, FollowupDecision, IntentResult, IntentType, ReplyCandidate
from ..prompting.reply_prompt_builder import ReplyPromptBuilder
from ..services.knowledge_service import KnowledgeService
from ..services.llm_service import LLMService


class ReplyAgent:
    """V2 文本回复 Agent（知识库优先，低置信/未命中再走 LLM）。"""

    def __init__(self, knowledge_service: KnowledgeService, llm_service: LLMService, kb_high_confidence: float = 0.65):
        self.knowledge_service = knowledge_service
        self.llm_service = llm_service
        self.prompt_builder = ReplyPromptBuilder()
        self.kb_high_confidence = float(kb_high_confidence)

    def generate_candidate(
        self,
        context: ConversationContext,
        intent_result: IntentResult,
        followup_decision: FollowupDecision,
    ) -> ReplyCandidate:
        latest_text = str(context.latest_user_text or "").strip()
        address_candidate = self._try_address_routing(latest_text)
        if address_candidate is not None:
            return address_candidate

        kb_detail = self._from_knowledge_detail(latest_text)
        if self._should_skip_address_kb(latest_text, kb_detail):
            kb_detail = self._retry_care_kb_detail(latest_text) or {"matched": False}
        elif (not kb_detail.get("matched")) and self._is_care_query(latest_text):
            kb_detail = self._retry_care_kb_detail(latest_text) or kb_detail
        kb_meta = self._build_empty_kb_meta()

        if kb_detail.get("matched"):
            kb_meta = {
                "matched": True,
                "score": float(kb_detail.get("score", 0.0) or 0.0),
                "item_id": str(kb_detail.get("item_id", "") or ""),
                "mode": str(kb_detail.get("mode", "") or ""),
                "answer_index": -1,
                "answer_mode": "",
                "repeat_fallback": False,
            }
            answer_pick = self._pick_kb_answer(
                answers=[str(x) for x in (kb_detail.get("answers") or [])],
                fallback_answer=str(kb_detail.get("answer", "") or ""),
                state=context.state,
                item_id=kb_meta["item_id"],
            )
            kb_meta.update(
                {
                    "answer_index": int(answer_pick.get("answer_index", -1)),
                    "answer_mode": str(answer_pick.get("answer_mode", "")),
                    "repeat_fallback": bool(answer_pick.get("repeat_fallback", False)),
                }
            )
            if self._is_kb_high_confidence(kb_detail, kb_meta):
                return ReplyCandidate(
                    text=self._normalize_answer(str(answer_pick.get("text", ""))),
                    source="kb_direct",
                    kb_meta=kb_meta,
                    should_regenerate=False,
                )

            llm_text = self._from_llm(
                context=context,
                intent_result=intent_result,
                followup_decision=followup_decision,
                failure_reason="",
                generation_path="kb_low_conf_llm",
                kb_detail=kb_detail,
                kb_candidates=self._build_kb_candidates(context.latest_user_text),
            )
            return ReplyCandidate(
                text=llm_text,
                source="kb_low_conf_llm",
                kb_meta=kb_meta,
                should_regenerate=True,
                llm_context={
                    "generation_path": "kb_low_conf_llm",
                    "kb_detail": kb_detail,
                    "kb_candidates": self._build_kb_candidates(context.latest_user_text),
                },
            )

        if followup_decision.need_followup and followup_decision.followup_text:
            return ReplyCandidate(
                text=followup_decision.followup_text,
                source="followup",
                kb_meta=kb_meta,
                should_regenerate=False,
            )

        llm_text = self._from_llm(
            context=context,
            intent_result=intent_result,
            followup_decision=followup_decision,
            failure_reason="",
            generation_path="llm_no_hit",
            kb_detail=None,
            kb_candidates=self._build_kb_candidates(context.latest_user_text),
        )
        return ReplyCandidate(
            text=llm_text,
            source="llm_no_hit",
            kb_meta=kb_meta,
            should_regenerate=True,
            llm_context={
                "generation_path": "llm_no_hit",
                "kb_detail": None,
                "kb_candidates": self._build_kb_candidates(context.latest_user_text),
            },
        )

    def _try_address_routing(self, text: str) -> ReplyCandidate | None:
        if not text:
            return None
        if not hasattr(self.knowledge_service, "resolve_store_recommendation"):
            return None
        route = self.knowledge_service.resolve_store_recommendation(text)
        if not isinstance(route, dict):
            return None
        reason = str(route.get("reason", "") or "")
        if reason in ("", "unknown"):
            return None

        # 护理类问题优先按护理回答，不被地址路由抢走。
        if self._is_care_query(text) and not self._is_explicit_address_query(text):
            return None

        if reason == "shanghai_need_district":
            return ReplyCandidate(
                text="姐姐您在上海哪个区呀？我马上给您匹配最近门店（静安/人广/虹口/五角场/徐汇）🌹",
                source="address_route",
                kb_meta={
                    "matched": False,
                    "score": 0.0,
                    "item_id": "",
                    "mode": "address_route",
                    "answer_index": -1,
                    "answer_mode": "route",
                    "repeat_fallback": False,
                    "route_reason": reason,
                    "target_store": "unknown",
                },
                should_regenerate=False,
            )

        if reason in ("sh_district_map:闵行", "sh_district_map:长宁", "sh_district_map:虹口", "sh_district_map:杨浦", "sh_district_map:五角场",
                      "sh_district_map:黄浦", "sh_district_map:黄埔", "sh_district_map:人民广场", "sh_district_map:人广", "sh_district_map:徐汇",
                      "sh_district_map:静安", "sh_district_map:浦东", "sh_district_map:青浦", "sh_district_map:金山", "sh_district_map:崇明",
                      "sh_district_map:宝山", "sh_district_map:普陀", "sh_district_map:松江", "sh_district_map:嘉定", "sh_district_map:奉贤",
                      "beijing_all_district", "north_fallback_beijing", "jiangzhe_to_sh_renmin"):
            target_store = str(route.get("target_store", "") or "")
            store_display = {}
            if hasattr(self.knowledge_service, "get_store_display"):
                store_display = self.knowledge_service.get_store_display(target_store)
            store_name = str((store_display or {}).get("store_name", "") or "就近门店")
            return ReplyCandidate(
                text=f"姐姐这边建议您到{store_name}，我可以马上给您对接到店路线和预约时间😊",
                source="address_route",
                kb_meta={
                    "matched": False,
                    "score": 0.0,
                    "item_id": "",
                    "mode": "address_route",
                    "answer_index": -1,
                    "answer_mode": "route",
                    "repeat_fallback": False,
                    "route_reason": reason,
                    "target_store": target_store,
                },
                should_regenerate=False,
            )

        if reason == "out_of_coverage":
            return ReplyCandidate(
                text="姐姐不在上海也可以远程定制，我先按您的头围和需求给您做方案，再安排寄送😊",
                source="address_route",
                kb_meta={
                    "matched": False,
                    "score": 0.0,
                    "item_id": "",
                    "mode": "address_route",
                    "answer_index": -1,
                    "answer_mode": "route",
                    "repeat_fallback": False,
                    "route_reason": reason,
                    "target_store": "unknown",
                },
                should_regenerate=False,
            )
        return None

    def regenerate_with_feedback(
        self,
        context: ConversationContext,
        intent_result: IntentResult,
        followup_decision: FollowupDecision,
        failure_reason: str,
        previous_candidate: ReplyCandidate,
    ) -> ReplyCandidate:
        if not previous_candidate.should_regenerate:
            return previous_candidate

        llm_context = previous_candidate.llm_context or {}
        llm_text = self._from_llm(
            context=context,
            intent_result=intent_result,
            followup_decision=followup_decision,
            failure_reason=failure_reason,
            generation_path=str(llm_context.get("generation_path", previous_candidate.source) or previous_candidate.source),
            kb_detail=llm_context.get("kb_detail"),
            kb_candidates=llm_context.get("kb_candidates") or [],
        )
        return ReplyCandidate(
            text=llm_text,
            source=previous_candidate.source,
            kb_meta=previous_candidate.kb_meta,
            should_regenerate=True,
            llm_context=llm_context,
        )

    def fallback_by_intent(self, intent: IntentType) -> str:
        templates = {
            IntentType.PRE_SALES: "价格和方案需要按您的需求评估后确认。",
            IntentType.AFTER_SALES: "售后处理建议要按您的具体佩戴情况判断。",
            IntentType.GREETING: "我在，您直接说最想先解决的问题。",
            IntentType.CLOSING: "好的，后续有问题随时来找我。",
            IntentType.OTHER: "我先按您这句问题给到最直接的答复。",
        }
        return templates.get(intent, templates[IntentType.OTHER])

    def _from_knowledge_detail(self, user_text: str) -> Dict[str, Any]:
        detail = self.knowledge_service.find_answer_detail(user_text, threshold=self.kb_high_confidence)
        if not isinstance(detail, dict):
            return {"matched": False}
        return detail

    def _retry_care_kb_detail(self, text: str) -> Dict[str, Any]:
        if not self._is_care_query(text):
            return {"matched": False}
        probes = [
            self._strip_location_noise(text),
            "假发怎么清洗",
            "如何清洗护理",
            "清洗护理",
        ]
        for probe in probes:
            probe_text = str(probe or "").strip()
            if not probe_text:
                continue
            detail = self.knowledge_service.find_answer_detail(probe_text, threshold=0.45)
            if isinstance(detail, dict) and detail.get("matched") and self._is_valid_care_detail(detail):
                return detail
        return {"matched": False}

    def _pick_kb_answer(
        self,
        answers: List[str],
        fallback_answer: str,
        state: Dict[str, Any],
        item_id: str,
    ) -> Dict[str, Any]:
        options = [str(x).strip() for x in (answers or []) if str(x).strip()]
        if not options and fallback_answer:
            options = [fallback_answer.strip()]
        if not options:
            return {
                "text": "",
                "answer_index": -1,
                "answer_mode": "none",
                "repeat_fallback": False,
            }

        seen_map = state.get("v2_kb_seen_count_by_item", {}) if isinstance(state, dict) else {}
        if not isinstance(seen_map, dict):
            seen_map = {}
        seen_count = int(seen_map.get(item_id, 0) or 0)

        recent_hashes = state.get("v2_recent_kb_answer_hashes", []) if isinstance(state, dict) else []
        if not isinstance(recent_hashes, list):
            recent_hashes = []

        if seen_count <= 0:
            return {
                "text": options[0],
                "answer_index": 0,
                "answer_mode": "first",
                "repeat_fallback": False,
            }

        candidate_indices = list(range(1, len(options))) + ([0] if options else [])
        non_repeat_indices = [
            idx for idx in candidate_indices if self._hash_text(options[idx]) not in set(str(x) for x in recent_hashes)
        ]
        if non_repeat_indices:
            selected_idx = random.choice(non_repeat_indices)
            return {
                "text": options[selected_idx],
                "answer_index": selected_idx,
                "answer_mode": "random",
                "repeat_fallback": False,
            }

        selected_idx = random.choice(candidate_indices)
        return {
            "text": options[selected_idx],
            "answer_index": selected_idx,
            "answer_mode": "random",
            "repeat_fallback": True,
        }

    def _from_llm(
        self,
        context: ConversationContext,
        intent_result: IntentResult,
        followup_decision: FollowupDecision,
        failure_reason: str,
        generation_path: str,
        kb_detail: Dict[str, Any] | None,
        kb_candidates: List[Dict[str, Any]] | None,
    ) -> str:
        prompt = self.prompt_builder.build(
            context=context,
            intent_result=intent_result,
            followup_decision=followup_decision,
            failure_reason=failure_reason,
            generation_path=generation_path,
            kb_detail=kb_detail,
            kb_candidates=kb_candidates or [],
        )
        self.llm_service.set_system_prompt(prompt)
        ok, reply = self.llm_service.generate_reply_sync(
            user_message=context.latest_user_text,
            conversation_history=context.history,
        )
        if not ok:
            return self.fallback_by_intent(intent_result.intent)
        return self._normalize_answer(str(reply or ""))

    def _build_kb_candidates(self, user_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query = self._normalize_for_match(user_text)
        if not query or not hasattr(self.knowledge_service, "repository"):
            return []
        repo = getattr(self.knowledge_service, "repository", None)
        if repo is None or not hasattr(repo, "get_all"):
            return []

        candidates: List[Dict[str, Any]] = []
        for item in repo.get_all()[:500]:
            question = str(getattr(item, "question", "") or "").strip()
            if not question:
                continue
            score = self._simple_overlap(query, self._normalize_for_match(question))
            if score <= 0:
                continue
            answer = str(getattr(item, "answer", "") or "").strip()
            if not answer:
                answers = getattr(item, "answers", []) or []
                answer = str(answers[0] if answers else "").strip()
            candidates.append(
                {
                    "item_id": str(getattr(item, "id", "") or ""),
                    "question": question,
                    "answer": answer,
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return candidates[:top_k]

    def _should_skip_address_kb(self, text: str, kb_detail: Dict[str, Any]) -> bool:
        if not isinstance(kb_detail, dict):
            return False
        if not kb_detail.get("matched"):
            return False
        intent = str(kb_detail.get("intent", "") or "").lower()
        if intent != "address":
            return False
        # 例如“不在上海如何清洗”这类护理诉求，不应命中纯地址回答。
        return self._is_care_query(text) and not self._is_explicit_address_query(text)

    def _is_explicit_address_query(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        address_keywords = ("地址", "门店", "在哪", "哪里", "怎么去", "到店", "路线", "哪个区")
        return any(k in normalized for k in address_keywords)

    def _is_care_query(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        care_keywords = ("清洗", "怎么洗", "护理", "保养", "头发乱", "打结", "炸毛")
        return any(k in normalized for k in care_keywords)

    def _strip_location_noise(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        noise_patterns = (
            r"不在上海",
            r"在上海",
            r"不在北京",
            r"在北京",
            r"异地",
            r"外地",
            r"如何",
            r"怎么",
            r"[？?！!，,。]",
        )
        cleaned = normalized
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    def _is_valid_care_detail(self, detail: Dict[str, Any]) -> bool:
        intent = str(detail.get("intent", "") or "").lower()
        question = str(detail.get("question", "") or "")
        care_terms = ("清洗", "护理", "保养", "打理")
        if intent not in ("care", "wearing"):
            return False
        return any(term in question for term in care_terms)

    def _simple_overlap(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.9
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    def _normalize_for_match(self, text: str) -> str:
        return re.sub(r"[\s，。！？、,.!?~]+", "", str(text or "").strip().lower())

    def _normalize_answer(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        cleaned = cleaned.replace("\n", "").replace("\r", "")
        return cleaned

    def _hash_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", str(text or ""))
        return normalized[:120]

    def _is_kb_high_confidence(self, kb_detail: Dict[str, Any], kb_meta: Dict[str, Any]) -> bool:
        score = float(kb_meta.get("score", 0.0) or 0.0)
        if score >= self.kb_high_confidence:
            return True

        mode = str(kb_meta.get("mode", "") or "")
        intent = str(kb_detail.get("intent", "") or "").lower()
        if mode in ("intent_hint", "contains", "normalized_contains", "normalized_exact") and intent in (
            "price",
            "wearing",
            "care",
            "address",
            "delivery_time",
        ):
            return score >= 0.45
        return False

    def _build_empty_kb_meta(self) -> Dict[str, Any]:
        return {
            "matched": False,
            "score": 0.0,
            "item_id": "",
            "mode": "none",
            "answer_index": -1,
            "answer_mode": "none",
            "repeat_fallback": False,
        }
