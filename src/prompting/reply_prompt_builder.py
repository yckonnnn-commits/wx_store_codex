"""已重构: reply_prompt_builder | V2 | 2026-03-01"""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain.chat_models import ConversationContext, FollowupDecision, IntentResult


class ReplyPromptBuilder:
    """构建回复生成提示，约束真人化短句风格。"""

    def build(
        self,
        context: ConversationContext,
        intent_result: IntentResult,
        followup_decision: FollowupDecision,
        failure_reason: str = "",
        generation_path: str = "",
        kb_detail: Dict[str, Any] | None = None,
        kb_candidates: List[Dict[str, Any]] | None = None,
    ) -> str:
        history_lines = [f"{item.get('role', '')}: {item.get('content', '')}" for item in context.history[-8:]]
        history_block = "\n".join(history_lines) if history_lines else "(无)"
        retry_line = f"上次失败原因：{failure_reason}\n" if failure_reason else ""
        path_line = f"生成路径：{generation_path}\n" if generation_path else ""
        followup_line = (
            f"追问策略：{followup_decision.reason}"
            if followup_decision.need_followup
            else "追问策略：不追问，直接给结论"
        )
        kb_line = ""
        if kb_detail:
            kb_line += (
                f"候选知识：question={kb_detail.get('question', '')}, "
                f"score={float(kb_detail.get('score', 0.0) or 0.0):.2f}, "
                f"answer={kb_detail.get('answer', '')}\n"
            )
        if kb_candidates:
            snippets = []
            for item in kb_candidates[:3]:
                snippets.append(
                    f"- q:{item.get('question', '')} | score:{float(item.get('score', 0.0) or 0.0):.2f} | a:{item.get('answer', '')}"
                )
            kb_line += "备选知识:\n" + "\n".join(snippets) + "\n"

        return (
            "你是私域客服，生成一条最终可发送回复。\n"
            "优先目标：准确回答用户当前问题，禁止答非所问。\n"
            "表达约束：只输出一条完整中文句子，可带或不带emoji；不能换行，不能省略号截断。\n"
            "若信息不确定，请明确边界，不要编造具体事实；不要输出机械模板空话。\n"
            f"意图：{intent_result.intent.value}，置信度：{intent_result.confidence:.2f}\n"
            f"{followup_line}\n"
            f"{path_line}"
            f"{kb_line}"
            f"最近对话：\n{history_block}\n"
            f"用户最新消息：{context.latest_user_text}\n"
            f"{retry_line}"
            "现在只输出最终回复文本，不要解释。"
        )
