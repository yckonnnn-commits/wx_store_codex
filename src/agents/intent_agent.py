"""已重构: intent_agent | V2 | 2026-03-01"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..domain.chat_models import IntentResult, IntentType


class IntentAgent:
    """V2 意图识别 Agent（规则最小集）。"""

    def __init__(self, rules_path: Path = Path("config") / "v2_agent_rules.json"):
        self.rules_path = rules_path
        self._intent_keywords: Dict[str, List[str]] = {}
        self.reload_rules()

    def reload_rules(self) -> None:
        default_rules = {
            "pre_sales": ["价格", "购买", "地址", "预约"],
            "after_sales": ["售后", "维修", "保养", "退换"],
            "greeting": ["你好", "在吗", "您好"],
            "closing": ["谢谢", "再见", "先这样"],
        }
        self._intent_keywords = default_rules
        if not self.rules_path.exists():
            return
        try:
            raw = json.loads(self.rules_path.read_text(encoding="utf-8"))
            loaded = raw.get("intent_keywords", {}) if isinstance(raw, dict) else {}
            if isinstance(loaded, dict):
                self._intent_keywords = {
                    str(k): [str(x) for x in (v or []) if str(x).strip()]
                    for k, v in loaded.items()
                }
        except Exception:
            self._intent_keywords = default_rules

    def analyze(self, latest_user_text: str, history: List[Dict[str, str]]) -> IntentResult:
        latest_text = (latest_user_text or "").strip().lower()
        if not latest_text:
            return IntentResult(intent=IntentType.OTHER, confidence=0.2, signals=["empty_input"])

        history_text = " ".join(item.get("content", "") for item in history[-2:]).strip().lower()
        scores = {
            IntentType.PRE_SALES: 0,
            IntentType.AFTER_SALES: 0,
            IntentType.GREETING: 0,
            IntentType.CLOSING: 0,
        }
        signals: List[str] = []

        mapping = {
            "pre_sales": IntentType.PRE_SALES,
            "after_sales": IntentType.AFTER_SALES,
            "greeting": IntentType.GREETING,
            "closing": IntentType.CLOSING,
        }

        for key, intent_type in mapping.items():
            for keyword in self._intent_keywords.get(key, []):
                kw = keyword.lower()
                if not kw:
                    continue
                if kw in latest_text:
                    scores[intent_type] += 3
                    signals.append(f"{intent_type.value}:{keyword}")
                    continue
                if history_text and kw in history_text:
                    scores[intent_type] += 1

        scores = self._boost_by_pattern(latest_text, scores=scores, signals=signals)
        best_intent = max(scores, key=lambda x: scores[x])
        best_score = scores[best_intent]
        if best_score <= 0:
            return IntentResult(intent=IntentType.OTHER, confidence=0.35, signals=["no_keyword_match"])

        confidence = min(0.95, 0.35 + best_score * 0.1)
        return IntentResult(intent=best_intent, confidence=confidence, signals=signals[:6])

    def _boost_by_pattern(
        self,
        latest_text: str,
        scores: Dict[IntentType, int],
        signals: List[str],
    ) -> Dict[IntentType, int]:
        purchase_or_product = ("材质", "价格", "多少钱", "怎么买", "门店", "地址", "在北京", "在上海", "会掉吗", "怎么戴")
        after_sales_care = ("清洗", "怎么洗", "洗", "护理", "保养", "维修", "返修", "佩戴了", "戴了多久")
        greeting_words = ("你好", "您好", "哈喽", "在吗")
        closing_words = ("谢谢", "再见", "拜拜", "先这样")

        if any(token in latest_text for token in purchase_or_product):
            scores[IntentType.PRE_SALES] += 2
            signals.append("pre_sales:pattern_hit")
        if any(token in latest_text for token in after_sales_care):
            scores[IntentType.AFTER_SALES] += 2
            signals.append("after_sales:pattern_hit")
        if any(token in latest_text for token in greeting_words):
            scores[IntentType.GREETING] += 2
            signals.append("greeting:pattern_hit")
        if any(token in latest_text for token in closing_words):
            scores[IntentType.CLOSING] += 2
            signals.append("closing:pattern_hit")

        if re.search(r"(怎么|如何).*(戴|清洗|洗|护理)", latest_text):
            scores[IntentType.AFTER_SALES] += 2
            signals.append("after_sales:howto_pattern")
        return scores
