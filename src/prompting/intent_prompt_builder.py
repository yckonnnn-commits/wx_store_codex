"""已重构: intent_prompt_builder | V2 | 2026-03-01"""

from __future__ import annotations

from typing import Dict, List


class IntentPromptBuilder:
    """构建意图识别提示词（当前 V2 先用于可观测日志与后续模型化）。"""

    def build(self, latest_user_text: str, history: List[Dict[str, str]]) -> str:
        clipped_history = history[-6:]
        history_lines = [f"{item.get('role', '')}: {item.get('content', '')}" for item in clipped_history]
        history_block = "\n".join(history_lines) if history_lines else "(无)"
        return (
            "你是意图分类器，只能输出 pre_sales/after_sales/greeting/closing/other。\n"
            "结合最新消息和最近上下文做判断。\n"
            f"最近对话:\n{history_block}\n"
            f"最新用户消息: {latest_user_text}"
        )
