"""已重构: followup_policy_agent | V2 | 2026-03-01"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..domain.chat_models import FollowupDecision, IntentResult, IntentType


class FollowupPolicyAgent:
    """V2 追问策略 Agent：最多追问 2 轮。"""

    def __init__(self, rules_path: Path = Path("config") / "v2_agent_rules.json"):
        self.rules_path = rules_path
        self.max_rounds = 2
        self.pre_sales_geo_keywords: List[str] = []
        self.after_sales_detail_keywords: List[str] = []
        self.reload_rules()

    def reload_rules(self) -> None:
        self.max_rounds = 2
        self.pre_sales_geo_keywords = ["上海", "北京", "徐汇", "静安", "虹口", "五角场", "人广", "朝阳"]
        self.after_sales_detail_keywords = ["佩戴", "多久", "照片", "图片", "发尾", "刘海", "头顶"]
        if not self.rules_path.exists():
            return
        try:
            raw = json.loads(self.rules_path.read_text(encoding="utf-8"))
            section = raw.get("followup", {}) if isinstance(raw, dict) else {}
            self.max_rounds = int(section.get("max_rounds", self.max_rounds) or self.max_rounds)
            pre = section.get("pre_sales_missing_geo_keywords", self.pre_sales_geo_keywords)
            aft = section.get("after_sales_missing_detail_keywords", self.after_sales_detail_keywords)
            self.pre_sales_geo_keywords = [str(x) for x in pre if str(x).strip()]
            self.after_sales_detail_keywords = [str(x) for x in aft if str(x).strip()]
        except Exception:
            return

    def decide(
        self,
        intent_result: IntentResult,
        session_state: Dict[str, Any],
        latest_user_text: str,
    ) -> FollowupDecision:
        followup_count = int(session_state.get("followup_count", 0) or 0)
        if followup_count >= self.max_rounds:
            return FollowupDecision(need_followup=False, followup_text="", reason="followup_limit_reached")

        text = (latest_user_text or "").strip()
        if intent_result.intent == IntentType.PRE_SALES:
            if not self._needs_geo_for_pre_sales(text):
                return FollowupDecision(need_followup=False, followup_text="", reason="pre_sales_direct_answer")
            has_geo = any(keyword in text for keyword in self.pre_sales_geo_keywords)
            if not has_geo:
                return FollowupDecision(
                    need_followup=True,
                    followup_text="可以的，您现在在哪个城市呀？😊",
                    reason="missing_geo",
                )

        if intent_result.intent == IntentType.AFTER_SALES:
            if not self._needs_detail_for_after_sales(text):
                return FollowupDecision(need_followup=False, followup_text="", reason="after_sales_direct_answer")
            has_detail = any(keyword in text for keyword in self.after_sales_detail_keywords)
            if not has_detail:
                return FollowupDecision(
                    need_followup=True,
                    followup_text="可以的，您先说下佩戴多久了呢？😊",
                    reason="missing_after_sales_detail",
                )

        return FollowupDecision(need_followup=False, followup_text="", reason="direct_answer")

    def _needs_geo_for_pre_sales(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        geo_required_keywords = (
            "门店",
            "地址",
            "到店",
            "哪家店",
            "在哪里",
            "怎么去",
            "预约",
            "到哪",
        )
        non_geo_product_keywords = (
            "材质",
            "价格",
            "多少钱",
            "会掉",
            "怎么戴",
            "怎么清洗",
            "睡觉",
        )
        if any(keyword in normalized for keyword in non_geo_product_keywords):
            return False
        return any(keyword in normalized for keyword in geo_required_keywords)

    def _needs_detail_for_after_sales(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        quick_answer_keywords = ("清洗", "怎么洗", "睡觉", "会掉吗", "怎么戴", "能戴吗")
        complaint_keywords = ("掉发", "毛躁", "打结", "变形", "不舒服", "扎", "过敏", "返修", "维修")
        if any(keyword in normalized for keyword in complaint_keywords):
            return True
        if any(keyword in normalized for keyword in quick_answer_keywords):
            return False
        return True
