"""已重构: reply_style_guard | V2 | 2026-03-01"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from ..domain.chat_models import GuardReport


class ReplyStyleGuard:
    """V2 回复风格守卫（准确优先）。"""

    _HANZI_RE = re.compile(r"[\u4e00-\u9fff]")
    _EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    def __init__(self, rules_path: Path = Path("config") / "v2_agent_rules.json"):
        self.rules_path = rules_path
        self.standard_min_hanzi = 10
        self.standard_max_hanzi = 30
        self.light_min_hanzi = 8
        self.light_max_hanzi = 45
        self.truncated_tails: List[str] = ["然后", "所以", "如果", "并且"]
        self.reload_rules()

    def reload_rules(self) -> None:
        if not self.rules_path.exists():
            return
        try:
            raw = json.loads(self.rules_path.read_text(encoding="utf-8"))
            section = raw.get("style_guard", {}) if isinstance(raw, dict) else {}
            self.standard_min_hanzi = int(section.get("standard_min_hanzi", self.standard_min_hanzi) or self.standard_min_hanzi)
            self.standard_max_hanzi = int(section.get("standard_max_hanzi", self.standard_max_hanzi) or self.standard_max_hanzi)
            self.light_min_hanzi = int(section.get("light_min_hanzi", self.light_min_hanzi) or self.light_min_hanzi)
            self.light_max_hanzi = int(section.get("light_max_hanzi", self.light_max_hanzi) or self.light_max_hanzi)
            tails = section.get("truncated_tails", self.truncated_tails)
            self.truncated_tails = [str(x) for x in tails if str(x).strip()]
        except Exception:
            return

    def sanitize(self, text: str) -> Tuple[str, bool]:
        candidate = str(text or "").strip()
        original = candidate
        candidate = candidate.replace("\r", "").replace("\n", "")
        candidate = re.sub(r"\s+", " ", candidate).strip()

        emojis = self._EMOJI_RE.findall(candidate)
        if len(emojis) > 1:
            keep = emojis[-1]
            candidate = self._EMOJI_RE.sub("", candidate).strip() + keep
            emojis = [keep]

        if emojis:
            emoji = emojis[-1]
            if not candidate.endswith(emoji):
                candidate = self._EMOJI_RE.sub("", candidate).strip() + emoji

        body = candidate
        tail_emoji = ""
        if emojis and candidate.endswith(emojis[-1]):
            tail_emoji = emojis[-1]
            body = candidate[:-len(tail_emoji)].rstrip()

        if body and body[-1] not in ("。", "！", "？"):
            body = body + "。"

        candidate = body + tail_emoji
        return candidate, candidate != original

    def validate(
        self,
        text: str,
        mode: str = "standard",
        attempts: int = 0,
        failed_reasons: List[str] | None = None,
    ) -> GuardReport:
        candidate = (text or "").strip()
        reasons = list(failed_reasons or [])

        if not candidate:
            return GuardReport(False, "empty_reply", attempts=attempts, failed_reasons=reasons)
        if "\n" in candidate:
            return GuardReport(False, "contains_newline", attempts=attempts, failed_reasons=reasons)

        emojis = self._EMOJI_RE.findall(candidate)
        if len(emojis) > 1:
            return GuardReport(False, "emoji_count_invalid", attempts=attempts, failed_reasons=reasons)
        if len(emojis) == 1 and not candidate.endswith(emojis[-1]):
            return GuardReport(False, "emoji_not_at_tail", attempts=attempts, failed_reasons=reasons)

        body = candidate[:-len(emojis[-1])].rstrip() if emojis else candidate
        if not body:
            return GuardReport(False, "missing_main_text", attempts=attempts, failed_reasons=reasons)

        terminal_count = sum(body.count(mark) for mark in ("。", "！", "？"))
        if terminal_count != 1:
            return GuardReport(False, "sentence_count_invalid", attempts=attempts, failed_reasons=reasons)
        if body[-1] not in ("。", "！", "？"):
            return GuardReport(False, "missing_terminal", attempts=attempts, failed_reasons=reasons)

        main_body = body[:-1].strip()
        if any(main_body.endswith(tail) for tail in self.truncated_tails):
            return GuardReport(False, "truncated_tail", attempts=attempts, failed_reasons=reasons)

        hanzi_count = len(self._HANZI_RE.findall(main_body))
        min_hanzi, max_hanzi = self._resolve_hanzi_range(mode)
        if hanzi_count < min_hanzi or hanzi_count > max_hanzi:
            return GuardReport(False, "hanzi_length_out_of_range", attempts=attempts, failed_reasons=reasons)

        if "..." in body or "……" in body:
            return GuardReport(False, "ellipsis_not_allowed", attempts=attempts, failed_reasons=reasons)

        return GuardReport(True, "passed", attempts=attempts, failed_reasons=reasons)

    def _resolve_hanzi_range(self, mode: str) -> Tuple[int, int]:
        if mode == "light":
            return self.light_min_hanzi, self.light_max_hanzi
        return self.standard_min_hanzi, self.standard_max_hanzi
