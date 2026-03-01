"""已重构: chat_models | V2 | 2026-03-01"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class IntentType(str, Enum):
    PRE_SALES = "pre_sales"
    AFTER_SALES = "after_sales"
    GREETING = "greeting"
    CLOSING = "closing"
    OTHER = "other"


@dataclass
class ConversationContext:
    session_id: str
    user_name: str
    latest_user_text: str
    history: List[Dict[str, str]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    signals: List[str] = field(default_factory=list)


@dataclass
class FollowupDecision:
    need_followup: bool
    followup_text: str
    reason: str


@dataclass
class GuardReport:
    passed: bool
    reason: str
    attempts: int = 0
    failed_reasons: List[str] = field(default_factory=list)


@dataclass
class FinalReply:
    text: str
    intent: IntentType
    source: str
    guard_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplyCandidate:
    text: str
    source: str
    kb_meta: Dict[str, Any] = field(default_factory=dict)
    should_regenerate: bool = True
    llm_context: Dict[str, Any] = field(default_factory=dict)
