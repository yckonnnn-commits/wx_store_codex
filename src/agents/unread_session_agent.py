"""已重构: unread_session_agent | V2 | 2026-03-01"""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain.chat_models import ConversationContext


class UnreadSessionAgent:
    """V2 未读会话上下文标准化 Agent。"""

    def build_context(
        self,
        session_id: str,
        user_name: str,
        latest_user_text: str,
        history: List[Dict[str, str]],
        state: Dict[str, Any],
    ) -> ConversationContext:
        return ConversationContext(
            session_id=session_id,
            user_name=user_name,
            latest_user_text=(latest_user_text or "").strip(),
            history=history,
            state=state or {},
        )
