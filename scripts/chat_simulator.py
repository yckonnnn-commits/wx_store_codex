#!/usr/bin/env python3
"""已重构: chat_simulator | V2 | 2026-03-01"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.chat_orchestrator import build_v2_orchestrator
from src.core.private_cs_agent import CustomerServiceAgent
from src.data.config_manager import ConfigManager
from src.data.knowledge_repository import KnowledgeRepository
from src.data.memory_store import MemoryStore
from src.domain.chat_models import ConversationContext
from src.services.knowledge_service import KnowledgeService
from src.services.llm_service import LLMService
from src.utils.constants import ENV_FILE, KNOWLEDGE_BASE_FILE, MODEL_SETTINGS_FILE


class StubLLMService:
    """规则联调时的本地占位 LLM，避免真实 API 调用。"""

    def __init__(self, fixed_reply: str = "可以的，我马上给您安排方案。😊"):
        self._prompt = ""
        self._fixed_reply = fixed_reply

    def set_system_prompt(self, prompt: str):
        self._prompt = prompt or ""

    def generate_reply_sync(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        del user_message, conversation_history
        return True, self._fixed_reply

    def get_current_model_name(self) -> str:
        return "StubLLM"


def build_services(no_llm: bool, stub_reply: str, sim_data_dir: Path):
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    convo_dir = sim_data_dir / "conversations"
    convo_dir.mkdir(parents=True, exist_ok=True)

    config_manager = ConfigManager(config_file=MODEL_SETTINGS_FILE, env_file=ENV_FILE)
    repository = KnowledgeRepository(data_file=KNOWLEDGE_BASE_FILE)
    knowledge_service = KnowledgeService(repository, address_config_path=Path("config") / "address.json")
    llm_service = StubLLMService(stub_reply) if no_llm else LLMService(config_manager)
    memory_store = MemoryStore(sim_data_dir / "agent_memory.json")
    return knowledge_service, llm_service, memory_store, convo_dir


def build_legacy_agent(knowledge_service, llm_service, memory_store, convo_dir: Path) -> CustomerServiceAgent:
    return CustomerServiceAgent(
        knowledge_service=knowledge_service,
        llm_service=llm_service,
        memory_store=memory_store,
        images_dir=Path("images"),
        image_categories_path=Path("config") / "image_categories.json",
        system_prompt_doc_path=Path("docs") / "system_prompt_private_ai_customer_service.md",
        playbook_doc_path=Path("docs") / "private_ai_customer_service_playbook.md",
        reply_templates_path=Path("config") / "reply_templates.json",
        media_whitelist_path=Path("config") / "media_whitelist.json",
        conversation_log_dir=convo_dir,
    )


def _append_session_event(
    session_log_file: Path,
    session_id: str,
    user_id_hash: str,
    event_type: str,
    payload: Dict[str, Any],
    reply_source: str = "",
    rule_id: str = "",
    model_name: str = "",
) -> None:
    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_id_hash": user_id_hash,
        "event_type": event_type,
        "reply_source": reply_source,
        "rule_id": rule_id,
        "model_name": model_name,
        "payload": payload,
    }
    session_log_file.parent.mkdir(parents=True, exist_ok=True)
    with session_log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="客服 Agent 命令行仿真器")
    parser.add_argument("-m", "--message", help="单次测试消息；不传则进入交互模式")
    parser.add_argument("--session-id", default="sim_session", help="会话 ID（默认 sim_session）")
    parser.add_argument("--user-name", default="sim_user", help="用户名（默认 sim_user）")
    parser.add_argument("--no-llm", action="store_true", help="禁用真实 LLM，使用本地占位回复")
    parser.add_argument("--stub-reply", default="可以的，我马上给您安排方案。😊", help="--no-llm 时的固定回复")
    parser.add_argument(
        "--sim-data-dir",
        default="data/simulator",
        help="仿真数据目录（记忆和会话日志），默认 data/simulator",
    )
    parser.add_argument(
        "--agent-mode",
        choices=("legacy", "v2"),
        default="v2",
        help="策略模式：legacy 或 v2，默认 v2",
    )
    args = parser.parse_args()

    sim_data_dir = Path(args.sim_data_dir)
    knowledge_service, llm_service, memory_store, convo_dir = build_services(
        no_llm=bool(args.no_llm),
        stub_reply=str(args.stub_reply or ""),
        sim_data_dir=sim_data_dir,
    )
    history: List[Dict[str, str]] = []
    session_log_file = sim_data_dir / "conversations" / f"{args.session_id}.jsonl"

    if args.agent_mode == "legacy":
        agent = build_legacy_agent(knowledge_service, llm_service, memory_store, convo_dir)
        user_hash = agent._hash_user(args.user_name or args.session_id)  # noqa: SLF001

        def run_once(user_text: str) -> None:
            text = (user_text or "").strip()
            if not text:
                return
            _append_session_event(
                session_log_file=session_log_file,
                session_id=args.session_id,
                user_id_hash=user_hash,
                event_type="user_message",
                payload={"text": text},
            )
            decision = agent.decide(
                session_id=args.session_id,
                user_name=args.user_name,
                latest_user_text=text,
                conversation_history=history,
            )
            extra_video = agent.mark_reply_sent(args.session_id, args.user_name, decision.reply_text)
            media_queue = list(decision.media_items or [])
            if extra_video:
                media_queue.append(extra_video)
            triggered_types = [str(item.get("type", "")) for item in media_queue if isinstance(item, dict)]

            _append_session_event(
                session_log_file=session_log_file,
                session_id=args.session_id,
                user_id_hash=user_hash,
                event_type="assistant_reply",
                reply_source=decision.reply_source,
                rule_id=decision.rule_id,
                model_name=decision.llm_model,
                payload={
                    "text": decision.reply_text,
                    "round_media_sent_types": triggered_types,
                },
            )
            print_payload(
                {
                    "mode": "legacy",
                    "reply_source": decision.reply_source,
                    "intent": decision.intent,
                    "route_reason": decision.route_reason,
                    "rule_id": decision.rule_id,
                    "media_plan": decision.media_plan,
                    "triggered_types_this_round": triggered_types,
                    "reply_text": decision.reply_text,
                }
            )
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": decision.reply_text})
            if len(history) > 20:
                del history[:-20]

    else:
        orchestrator = build_v2_orchestrator(knowledge_service=knowledge_service, llm_service=llm_service)
        user_hash = f"sim_{args.user_name}"

        def run_once(user_text: str) -> None:
            text = (user_text or "").strip()
            if not text:
                return
            session_state = memory_store.get_session_state(args.session_id, user_hash=user_hash)
            user_state = memory_store.get_user_state(user_hash)
            context = ConversationContext(
                session_id=args.session_id,
                user_name=args.user_name,
                latest_user_text=text,
                history=history,
                state={
                    "last_intent": str(session_state.get("v2_last_intent", "other")),
                    "followup_count": int(session_state.get("v2_followup_count", 0) or 0),
                    "recent_reply_hashes": list(user_state.get("recent_reply_hashes", []) or []),
                    "v2_kb_seen_count_by_item": dict(session_state.get("v2_kb_seen_count_by_item", {}) or {}),
                    "v2_recent_kb_answer_hashes": list(session_state.get("v2_recent_kb_answer_hashes", []) or []),
                },
            )
            final_reply = orchestrator.process(context)
            next_followup_count = int(session_state.get("v2_followup_count", 0) or 0) + 1 if final_reply.source == "followup" else 0
            guard_report = final_reply.guard_report if isinstance(final_reply.guard_report, dict) else {}
            kb_meta = guard_report.get("kb", {}) if isinstance(guard_report.get("kb", {}), dict) else {}
            kb_item_id = str(kb_meta.get("item_id", "") or "")
            seen_map = dict(session_state.get("v2_kb_seen_count_by_item", {}) or {})
            recent_kb_hashes = list(session_state.get("v2_recent_kb_answer_hashes", []) or [])
            if bool(kb_meta.get("matched", False)) and kb_item_id:
                seen_map[kb_item_id] = int(seen_map.get(kb_item_id, 0) or 0) + 1
                recent_kb_hashes.append(final_reply.text[-40:])
            memory_store.update_session_state(
                args.session_id,
                user_hash=user_hash,
                updates={
                    "v2_last_intent": final_reply.intent.value,
                    "v2_followup_count": next_followup_count,
                    "v2_kb_seen_count_by_item": seen_map,
                    "v2_recent_kb_answer_hashes": recent_kb_hashes[-40:],
                },
            )
            recent = list(user_state.get("recent_reply_hashes", []) or [])
            recent.append(final_reply.text[-12:])
            memory_store.update_user_state(user_hash, updates={"recent_reply_hashes": recent[-20:]})
            memory_store.save()

            _append_session_event(
                session_log_file=session_log_file,
                session_id=args.session_id,
                user_id_hash=user_hash,
                event_type="assistant_reply",
                reply_source=f"v2_{final_reply.source}",
                rule_id="V2_ORCHESTRATOR",
                model_name=llm_service.get_current_model_name() if hasattr(llm_service, "get_current_model_name") else "",
                payload={
                    "text": final_reply.text,
                    "intent": final_reply.intent.value,
                    "guard_report": final_reply.guard_report,
                },
            )
            print_payload(
                {
                    "mode": "v2",
                    "reply_source": f"v2_{final_reply.source}",
                    "intent": final_reply.intent.value,
                    "guard_report": final_reply.guard_report,
                    "kb_matched": bool(kb_meta.get("matched", False)),
                    "kb_score": float(kb_meta.get("score", 0.0) or 0.0),
                    "kb_item_id": str(kb_meta.get("item_id", "") or ""),
                    "kb_answer_index": int(
                        kb_meta.get("answer_index", -1)
                        if kb_meta.get("answer_index", None) is not None
                        else -1
                    ),
                    "kb_answer_mode": str(kb_meta.get("answer_mode", "none") or "none"),
                    "generation_path": str(guard_report.get("generation_path", final_reply.source) or final_reply.source),
                    "guard_sanitized": bool(guard_report.get("sanitized", False)),
                    "reply_text": final_reply.text,
                }
            )
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": final_reply.text})
            if len(history) > 20:
                del history[:-20]

    if args.message:
        run_once(args.message)
        return 0

    print("进入交互模式。输入 /exit 退出，输入 /reset 清空当前会话上下文。")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not text:
            continue
        if text in ("/exit", "exit", "quit"):
            print("已退出。")
            return 0
        if text == "/reset":
            history.clear()
            print("当前会话上下文已清空。")
            continue
        run_once(text)


if __name__ == "__main__":
    raise SystemExit(main())
