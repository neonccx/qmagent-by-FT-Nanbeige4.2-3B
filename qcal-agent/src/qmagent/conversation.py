"""Chat routing: proposals are data, never executable model output."""

import json
import re

from .contracts import TOOLS


TARGET_ALIASES = {
    "sq.s21": ("s21", "resonator", "谐振腔", "谐振器"),
    "sq.s21_power2d": ("s21power2d", "power2d", "功率二维", "功率扫描"),
    "sq.s21_zpa2d": ("s21zpa2d", "zpa2d", "zpa", "磁通二维", "磁通扫描"),
    "sq.spectroscopy": ("spectroscopy", "spectrum", "能谱", "频谱"),
    "sq.piamp": ("piamp", "rabi", "拉比", "π脉冲", "pi脉冲"),
    "sq.ramsey_df": ("ramseydf", "ramsey", "拉姆齐"),
    "sq.t1": ("t1", "弛豫"),
    "sq.t2_echo": ("t2echo", "echo", "回波"),
    "sq.xeb": ("xeb",),
    "sq.iqraw": ("iqraw", "iqmeasurement", "iqmeasure", "iq测量", "iq采集", "iq分割", "iq"),
}


def resolve_experiment_target(text):
    """Resolve a user-facing experiment name without accepting arbitrary tools."""
    if not isinstance(text, str):
        return None
    normalized = re.sub(r"[\s_./，。？！?!:：-]", "", text).lower()
    matches = []
    for tool, aliases in TARGET_ALIASES.items():
        matches.extend((len(alias), tool) for alias in aliases if alias in normalized)
    if not matches:
        return None
    longest = max(size for size, _ in matches)
    winners = {tool for size, tool in matches if size == longest}
    return next(iter(winners)) if len(winners) == 1 else None


class ReplyStream:
    """Stream prose, withholding split action tags until the final reply is validated."""

    marker = "<qm_action>"

    def __init__(self, callback):
        self.callback = callback
        self.pending = self.emitted = ""
        self.hidden = False
        self.received = 0

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError("Invalid text delta")
        self.received += len(text)
        if self.received > 65536:
            raise ValueError("Model stream exceeds reply limit")
        if self.hidden:
            return
        self.pending += text
        if self.marker in self.pending:
            self.pending = self.pending.split(self.marker, 1)[0]
            self.hidden = True
        hold = 0
        if not self.hidden:
            for size in range(1, len(self.marker)):
                if self.pending.endswith(self.marker[:size]):
                    hold = size
        visible = self.pending[:-hold] if hold else self.pending
        visible = visible.rstrip()
        self.pending = self.pending[len(visible):]
        if not self.emitted:
            visible = visible.lstrip()
        self._emit(visible)

    def _emit(self, text):
        if text:
            self.emitted += text
            self.callback(text)

    def finish(self, message):
        # The final envelope is authoritative. A tokenizer may revise whitespace
        # in its full decode; let the UI reconcile instead of losing the reply.
        if not message.startswith(self.emitted):
            return False
        self._emit(message[len(self.emitted):])
        return True


def validate_message(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("Question must contain 1-8000 characters")
    if re.search(r"(?i)(密码|password|api[_ -]?key|access[_ -]?token|验证码)\s*(?:是|[:=：])\s*\S+", text):
        raise ValueError("Do not enter credentials here. Content was not recorded or sent to the model. Use system SSH for login.")


def validate_proposal(value):
    if not isinstance(value, dict) or set(value) - {"command", "limit", "target", "report"}:
        raise ValueError("Invalid proposed command")
    command = value.get("command")
    if command not in {"plan", "step", "run", "status", "report", "goal", "diagnostic"}:
        raise ValueError("Unlisted proposed command")
    if "limit" in value and (command != "run" or type(value["limit"]) is not int or not 1 <= value["limit"] <= 1000):
        raise ValueError("Invalid proposed run limit")
    if command in {"goal", "diagnostic"}:
        if value.get("target") not in TOOLS or type(value.get("report", False)) is not bool:
            raise ValueError("Invalid experiment goal")
    elif "target" in value or "report" in value:
        raise ValueError("Only goal/diagnostic proposals accept target/report")
    return dict(value)


def parse_reply(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 65536:
        raise ValueError("Model returned an empty/oversized reply")
    marker = "<qm_action>"
    if marker not in text:
        return text.strip(), None, None
    match = re.fullmatch(r"(.*?)\s*<qm_action>(.*?)</qm_action>\s*", text, re.S)
    if not match or text.count(marker) != 1:
        return text.strip(), None, "Malformed operation tag; no tool called"
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate proposal key")
            value[key] = item
        return value
    try:
        proposal = validate_proposal(json.loads(match[2], object_pairs_hook=unique))
        return match[1].strip() or "I propose the following operation, subject to your confirmation.", proposal, None
    except (ValueError, TypeError, RecursionError):
        return match[1].strip() or "The model proposed an invalid operation.", None, "Proposed operation is outside the allowlist; not executed"


def explicit_intent(text):
    """Fast paths for common requests; other sentences go to the actual LLM."""
    normalized = re.sub(r"[\s，。？！?!]", "", text)
    if normalized.lower() in {"run", "plan", "step", "status", "report"}:
        return {"command": normalized.lower()}
    if normalized in {"状态", "当前状态", "现在校准到哪了", "现在校准到哪一步了", "校准到哪一步了", "看看当前状态"}:
        return {"command": "status"}
    if re.fullmatch(r"(?:请|帮我|请帮我|我想|麻烦你)?(?:开始|继续)(?:进行)?校准(?:这个比特|这个量子比特|一下)?", normalized):
        return {"command": "run"}
    if normalized in {"下一步", "执行下一步", "帮我执行下一步", "运行一步", "执行一步"}:
        return {"command": "step"}
    if normalized in {"看看下一步计划", "给我下一步计划", "帮我制定下一步计划"}:
        return {"command": "plan"}
    target = resolve_experiment_target(text)
    action_requested = (re.search(r"做|进行|执行|运行|测量|采集|完成|生成|绘制|画", text)
                        or re.search(r"(?i)\b(run|perform|execute|measure|acquire|complete|generate|plot)\b", text))
    if target and action_requested:
        wants_report = bool(re.search(r"(?i)(report|报告|图|绘制|画|plot)", text))
        goal_requested = bool(re.search(r"(?i)(运行到|校准到|直到|完成.*校准|through|until|calibrate.*through)", text))
        return {"command": "goal" if goal_requested else "diagnostic",
                "target": target, "report": wants_report}
    return None
