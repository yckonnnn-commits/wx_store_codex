#!/usr/bin/env python3
"""已重构: v2_replay_eval | V2 | 2026-03-01"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.chat_orchestrator import build_v2_orchestrator
from src.data.config_manager import ConfigManager
from src.data.knowledge_repository import KnowledgeRepository
from src.services.knowledge_service import KnowledgeService
from src.utils.constants import ENV_FILE, KNOWLEDGE_BASE_FILE, MODEL_SETTINGS_FILE


class StubLLMService:
    def __init__(self, fixed_reply: str = "我理解您的问题，建议按实际佩戴情况评估后确认。"):
        self.fixed_reply = fixed_reply
        self.prompt = ""

    def set_system_prompt(self, prompt: str):
        self.prompt = prompt

    def generate_reply_sync(self, user_message: str, conversation_history=None):
        del user_message, conversation_history
        return True, self.fixed_reply

    def get_current_model_name(self) -> str:
        return "StubLLM"


def load_messages(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        result: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "") or row.get("message", "") or "").strip()
            if text:
                result.append(text)
        return result

    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
                elif isinstance(item, dict):
                    text = str(item.get("text", "") or item.get("message", "") or "").strip()
                    if text:
                        result.append(text)
            return result

    return [line.strip() for line in raw.splitlines() if line.strip()]


def hanzi_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def is_template_blank(text: str) -> bool:
    blank_markers = ("说下重点", "马上处理", "我在呢", "告诉我最关心")
    return any(m in (text or "") for m in blank_markers)


def evaluate(messages: List[str], kb_threshold: float) -> Dict[str, Any]:
    config_manager = ConfigManager(config_file=MODEL_SETTINGS_FILE, env_file=ENV_FILE)
    repository = KnowledgeRepository(data_file=KNOWLEDGE_BASE_FILE)
    knowledge_service = KnowledgeService(repository, address_config_path=Path("config") / "address.json")
    llm = StubLLMService()
    orchestrator = build_v2_orchestrator(
        knowledge_service=knowledge_service,
        llm_service=llm,
        kb_high_confidence=kb_threshold,
    )

    history: List[Dict[str, str]] = []
    state: Dict[str, Any] = {
        "followup_count": 0,
        "v2_kb_seen_count_by_item": {},
        "v2_recent_kb_answer_hashes": [],
        "recent_reply_hashes": [],
    }

    results: List[Dict[str, Any]] = []
    for text in messages:
        from src.domain.chat_models import ConversationContext

        context = ConversationContext(
            session_id="replay_eval",
            user_name="replay_user",
            latest_user_text=text,
            history=history,
            state=state,
        )
        reply = orchestrator.process(context)
        guard = reply.guard_report if isinstance(reply.guard_report, dict) else {}
        kb = guard.get("kb", {}) if isinstance(guard.get("kb", {}), dict) else {}

        if bool(kb.get("matched", False)) and str(kb.get("item_id", "") or ""):
            item_id = str(kb.get("item_id"))
            seen = state.get("v2_kb_seen_count_by_item", {})
            if not isinstance(seen, dict):
                seen = {}
            seen[item_id] = int(seen.get(item_id, 0) or 0) + 1
            state["v2_kb_seen_count_by_item"] = seen
            recent = state.get("v2_recent_kb_answer_hashes", [])
            if not isinstance(recent, list):
                recent = []
            recent.append(reply.text[-40:])
            state["v2_recent_kb_answer_hashes"] = recent[-40:]

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply.text})
        if len(history) > 20:
            del history[:-20]

        results.append(
            {
                "user": text,
                "reply": reply.text,
                "source": reply.source,
                "generation_path": guard.get("generation_path", ""),
                "kb_matched": bool(kb.get("matched", False)),
                "kb_score": float(kb.get("score", 0.0) or 0.0),
                "hanzi": hanzi_count(reply.text),
                "template_blank": is_template_blank(reply.text),
            }
        )

    total = max(1, len(results))
    kb_direct = sum(1 for x in results if x["source"] == "kb_direct")
    kb_any = sum(1 for x in results if x["kb_matched"])
    llm_no_hit = sum(1 for x in results if x["source"] == "llm_no_hit")
    template_blank_count = sum(1 for x in results if x["template_blank"])
    len_pass = sum(1 for x in results if 8 <= int(x["hanzi"]) <= 45)

    return {
        "summary": {
            "total": len(results),
            "kb_direct_rate": kb_direct / total,
            "kb_matched_rate": kb_any / total,
            "llm_no_hit_rate": llm_no_hit / total,
            "length_pass_rate_8_45": len_pass / total,
            "template_blank_rate": template_blank_count / total,
        },
        "samples": results[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 回放评估脚本")
    parser.add_argument("--input", required=True, help="输入文件（txt/json/jsonl），每条一问")
    parser.add_argument("--output", default="data/replay_eval/v2_eval_report.json", help="输出报告路径")
    parser.add_argument("--kb-threshold", type=float, default=0.65, help="V2 KB 高置信阈值")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return 1

    messages = load_messages(input_path)
    report = evaluate(messages, kb_threshold=float(args.kb_threshold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    print(f"报告已输出: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
