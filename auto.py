#!/usr/bin/env python3
import hashlib
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

TASK = sys.argv[1]

# slug: first ~30 chars kebab-case + short hash for uniqueness
_slug = re.sub(r"[^\w\s-]", "", TASK[:30]).strip()
_slug = re.sub(r"[\s_]+", "-", _slug).lower().strip("-") or "task"
_hash = hashlib.md5(TASK.encode()).hexdigest()[:6]
PLAN = Path(f"plan-{_slug}-{_hash}.md")

# -- ANSI colors --
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


PROJECT_ROOT = str(Path(__file__).resolve().parent) + "/"


def short_path(p):
    """Shorten absolute paths to relative from project root."""
    return p.replace(PROJECT_ROOT, "") if isinstance(p, str) else p


def log(msg):
    if "Claude" in msg:
        color = ORANGE
    elif "Codex" in msg:
        color = CYAN
    else:
        color = ""
    print(f"\n{color}{BOLD}=== {msg} ==={RESET}", flush=True)


def _read_lines(fd, proc):
    """Yield complete lines from fd, non-blocking."""
    buf = ""
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                data = os.read(fd, 8192)
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line.strip()
            elif proc.poll() is not None:
                while True:
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if not ready:
                        break
                    data = os.read(fd, 8192)
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield line.strip()
                if buf.strip():
                    yield buf.strip()
                break
        except OSError:
            break


def run_codex(prompt, label=None):
    """Run codex exec with --json and display progress."""
    print(f"{DIM}[START] {label}{RESET}", flush=True)
    t0 = time.time()

    proc = subprocess.Popen(
        [
            "codex", "exec", "--full-auto", "--json",
            f"IMPORTANT: Do NOT run any git commands (git add, git commit, git push, git checkout, git reset, etc.).\n\n{prompt}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    fd = proc.stdout.fileno()
    result_text = ""
    total_tokens = 0

    for line in _read_lines(fd, proc):
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type", "")
        elapsed = time.time() - t0
        item = evt.get("item", {})
        itype = item.get("type", "")

        if etype == "item.completed":
            if itype == "reasoning":
                snippet = item.get("text", "")[:60].replace("\n", " ")
                print(f"  {DIM}[{elapsed:5.1f}s] thinking: {snippet}{RESET}", flush=True)

            elif itype == "agent_message":
                text = item.get("text", "")
                result_text = text
                # show first few lines
                for i, ln in enumerate(text.split("\n")):
                    if i >= 5:
                        print(f"  {DIM}  ...({len(text.split(chr(10)))} lines){RESET}", flush=True)
                        break
                    print(f"  {ln}", flush=True)

            elif itype == "command_execution":
                cmd_str = short_path(item.get("command", ""))
                exit_code = item.get("exit_code", "?")
                status = f"{GREEN}ok{RESET}" if exit_code == 0 else f"{YELLOW}exit {exit_code}{RESET}"
                print(
                    f"  {CYAN}[{elapsed:5.1f}s] exec ({status}{CYAN}): {cmd_str}{RESET}",
                    flush=True,
                )

            elif itype == "file_update":
                path = short_path(item.get("file_path", ""))
                print(
                    f"  {CYAN}[{elapsed:5.1f}s] file: {BOLD}{path}{RESET}",
                    flush=True,
                )

        elif etype == "item.started":
            if itype == "command_execution":
                cmd_str = short_path(item.get("command", ""))
                print(f"  {DIM}[{elapsed:5.1f}s] exec: {cmd_str}...{RESET}", flush=True)

        elif etype == "turn.completed":
            usage = evt.get("usage", {})
            total_tokens += usage.get("output_tokens", 0)

    proc.wait()

    dur = time.time() - t0
    print(
        f"\n  {GREEN}[DONE] {label} "
        f"({dur:.1f}s, {total_tokens} tokens){RESET}\n",
        flush=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with code {proc.returncode}")

    return result_text


def run_claude(prompt, label=None):
    """Run claude -p with stream-json and display progress."""
    print(f"{DIM}[START] {label}{RESET}", flush=True)
    t0 = time.time()

    cmd = [
        "claude", "-p",
        "--output-format", "stream-json", "--verbose",
        "--allowedTools", ALLOWED_TOOLS,
        "--disallowedTools", *DISALLOWED_TOOLS,
        "--", prompt,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    fd = proc.stdout.fileno()
    result_text = ""
    turn = 0

    for line in _read_lines(fd, proc):
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            print(line, flush=True)
            continue

        etype = evt.get("type", "")
        elapsed = time.time() - t0

        if etype == "system":
            model = evt.get("model", "?")
            print(f"  {DIM}model={model}{RESET}", flush=True)

        elif etype == "assistant":
            msg = evt.get("message", {})
            for block in msg.get("content", []):
                bt = block.get("type")
                if bt == "text":
                    text = block["text"]
                    print(f"  {text}", flush=True)
                elif bt == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    turn += 1
                    summary = ""
                    if name == "Bash":
                        summary = inp.get("description") or short_path(inp.get("command", ""))[:80]
                    elif name in ("Read", "Edit", "Write"):
                        summary = short_path(inp.get("file_path", ""))
                    elif name == "Glob":
                        summary = inp.get("pattern", "")
                    elif name == "Grep":
                        summary = inp.get("pattern", "")
                    else:
                        summary = short_path(json.dumps(inp))[:80]
                    print(
                        f"  {ORANGE}[{elapsed:5.1f}s] "
                        f"tool: {BOLD}{name}{RESET}{ORANGE} "
                        f"- {summary}{RESET}",
                        flush=True,
                    )
                elif bt == "thinking":
                    snippet = block.get("thinking", "")[:60].replace("\n", " ")
                    print(
                        f"  {DIM}[{elapsed:5.1f}s] thinking: {snippet}...{RESET}",
                        flush=True,
                    )

        elif etype == "user":
            # tool results from claude
            msg = evt.get("message", {})
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    is_err = block.get("is_error", False)
                    status = f"{YELLOW}error{RESET}" if is_err else f"{GREEN}ok{RESET}"
                    preview = short_path(content)[:120].replace("\n", " ") if isinstance(content, str) else ""
                    print(
                        f"  {DIM}[{elapsed:5.1f}s] result ({status}{DIM}): {preview}{RESET}",
                        flush=True,
                    )

        elif etype == "result":
            result_text = evt.get("result", "")
            cost = evt.get("total_cost_usd", 0)
            turns = evt.get("num_turns", 0)
            dur = evt.get("duration_ms", 0) / 1000
            print(
                f"\n  {GREEN}[DONE] {label} "
                f"({dur:.1f}s, {turns} turns, ${cost:.4f}){RESET}\n",
                flush=True,
            )

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with code {proc.returncode}")

    return result_text


print(f"TASK: {TASK}")
print(f"PLAN: {PLAN}")

# Step 1: Claude plan
log("Step 1/4: Claude generating plan")

ALLOWED_TOOLS = "Read,Edit,Write,Bash"
DISALLOWED_TOOLS = [
    "Bash(rm:*)",
    "Bash(rmdir:*)",
    "Bash(unlink:*)",
    "Bash(shred:*)",
    "Bash(dd:*)",
    "Bash(mkfs:*)",
    "Bash(git:*)",
]

claude_prompt = f"""\
Create plan.md for this task:
{TASK}
Rules:
- concrete steps
- exact files
- minimal scope
"""

plan = run_claude(claude_prompt, label="Claude Plan")

PLAN.write_text(plan)


# Step 2: Codex review
log("Step 2/4: Codex reviewing plan")

plan = run_codex(
    f"Review and improve this plan.md. Edit directly. "
    f"Return full updated plan.md.\n\n{PLAN.read_text()}",
    label="Codex Review",
)

PLAN.write_text(plan)


# Step 3: Claude implement
log("Step 3/4: Claude implementing")

run_claude(
    f"Implement this plan. Apply changes directly to files.\n\n{PLAN.read_text()}",
    label="Claude Implement",
)


# Step 4: Codex verify (git diff vs plan.md)
log("Step 4/4: Codex verifying")

diff = subprocess.run(
    ["git", "diff"], capture_output=True, text=True
).stdout

untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard"],
    capture_output=True, text=True,
).stdout

verify = run_codex(
    f"Verify that the following changes are consistent with the plan.\n"
    f"Report any missing steps, extra changes, or issues.\n"
    f"If everything is correct, say 'LGTM'.\n\n"
    f"## Plan\n{PLAN.read_text()}\n\n"
    f"## git diff\n```\n{diff}\n```\n\n"
    f"## New untracked files\n{untracked or '(none)'}\n",
    label="Codex Verify",
)

VERIFY_FILE = PLAN.with_name(PLAN.stem + "-verify.md")
VERIFY_FILE.write_text(verify)

print(f"\n{BOLD}Verification result ({VERIFY_FILE}):{RESET}")
print(verify)

log("COMPLETE")