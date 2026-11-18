"""Read-only adapters for agents besides Codex: Claude Code transcripts and magpie.

Neither format is a public API. Read only what the dashboard shows (folder,
title, last turn state, final answer), never tool output or reasoning, never
write, and fail per source: one unreadable agent must not hide the others.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from .data import CodexReader, Unavailable, display_title, match_root

HEAD_BYTES = 256 * 1024
TAIL_BYTES = 512 * 1024
# A Claude Code turn leaves no "in progress" flag. A transcript that has not
# been written for this long is reported as unknown, not as still running.
RUNNING_WINDOW = 1800
DONE_STOPS = {"end_turn", "stop_sequence", "max_tokens"}
INTERRUPTED = "[Request interrupted by user"
SESSION_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{7,63}$")
FRACTION = re.compile(r"\.(\d+)")
SOURCES = ("codex", "claude", "magpie")
AGENT_LABELS = {
    "codex": "Codex", "claude": "Claude Code", "claude-desktop": "Claude Desktop", "antigravity": "Antigravity",
    "opencode": "OpenCode", "pi": "Pi", "cursor": "Cursor", "droid": "Droid", "gemini": "Gemini CLI",
    "hermes": "Hermes Agent", "zcode": "ZCode", "dsh": "DeepSeek Harness", "goose": "Goose", "copilot": "Copilot",
}


class Absent(Unavailable):
    """The agent is not installed on this computer. Not an error to report."""


def enabled_sources():
    raw = os.environ.get("PULSE_SOURCES")
    names = [s.strip() for s in raw.split(",")] if raw is not None else list(SOURCES)
    return [s for s in names if s in SOURCES]


def window_seconds():
    try:
        return max(1.0, float(os.environ.get("PULSE_DAYS", "30"))) * 86400
    except ValueError:
        return 30 * 86400


def epoch(value):
    """ISO 8601 (Z, offsets, 2–9 fraction digits) → seconds; Python 3.9 compatible."""
    if not isinstance(value, str) or not value:
        return None
    text = FRACTION.sub(lambda m: "." + m.group(1).ljust(6, "0")[:6], value.strip().replace("Z", "+00:00"), count=1)
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def magpie_home():
    return Path(os.environ.get("PULSE_MAGPIE_HOME") or Path.home() / ".config" / "magpie")


# ---------- Claude Code ----------

def claude_dirs():
    """Claude Code config folders: CLAUDE_CONFIG_DIR, ~/.claude, and magpie's per-account folders."""
    raw = os.environ.get("PULSE_CLAUDE_DIRS")
    if raw is not None:
        candidates = [p for p in raw.split(os.pathsep) if p]
    else:
        candidates = [os.environ.get("CLAUDE_CONFIG_DIR") or "", str(Path.home() / ".claude")]
        accounts = magpie_home() / "claude-accounts"
        if accounts.is_dir():
            candidates += sorted(str(p) for p in accounts.iterdir() if p.is_dir())
    seen, out = set(), []
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        real = os.path.realpath(path)
        if real in seen or not (path / "projects").is_dir():
            continue
        seen.add(real)
        out.append(path)
    return out


def default_claude_dir():
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()


# Records the dashboard reads. Transcripts are mostly API request/response
# bookkeeping; skipping those lines before JSON decoding keeps a cold scan of
# thousands of sessions to about a second.
WANTED = re.compile(rb'"type"\s*:\s*"(?:user|assistant|ai-title|custom-title|summary|last-prompt)"')


def records(blob, reverse=False):
    """Wanted records of a byte range, lazily, newest first when reverse."""
    lines = blob.split(b"\n")
    for line in reversed(lines) if reverse else lines:
        if not WANTED.search(line):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # partial line at a read boundary, or one being appended
        if isinstance(record, dict):
            yield record


def read_blobs(path, size):
    """The head and the tail of a transcript; large files are never read whole."""
    with open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
        tail = head
        if size > HEAD_BYTES:
            fh.seek(max(HEAD_BYTES, size - TAIL_BYTES))
            tail = fh.read()
    return head, tail


def is_main(record):
    return record.get("type") in ("user", "assistant") and not record.get("isSidechain") and not record.get("isMeta")


def content_of(record):
    message = record.get("message")
    return message.get("content") if isinstance(message, dict) else None


def text_of(record):
    content = content_of(record)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def is_prompt(record):
    """A message the user typed, not a tool result returned to the model."""
    if record.get("type") != "user" or not is_main(record):
        return False
    content = content_of(record)
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return False
    return bool(text_of(record).strip())


TITLES = (("custom-title", "customTitle"), ("ai-title", "aiTitle"), ("summary", "summary"))


def tail_facts(blob):
    """Newest first: the last main message, the last typed prompt, the latest titles."""
    last = prompt = None
    titles, last_prompt = {}, ""
    for record in records(blob, reverse=True):
        kind = record.get("type")
        field = dict(TITLES).get(kind)
        if field:
            value = record.get(field)
            if kind not in titles and isinstance(value, str) and value.strip():
                titles[kind] = value
        elif kind == "last-prompt":
            last_prompt = last_prompt or (record.get("lastPrompt") if isinstance(record.get("lastPrompt"), str) else "")
        elif is_main(record):
            last = last or record
            prompt = prompt or (record if is_prompt(record) else None)
        if last and prompt and titles:
            break
    return last, prompt, titles, last_prompt


def head_facts(blob):
    """Oldest first: the session id, its folder and the first typed prompt."""
    session_id = cwd = first = None
    for record in records(blob):
        if not is_main(record):
            continue
        session_id = session_id or (record.get("sessionId") if isinstance(record.get("sessionId"), str) else None)
        cwd = cwd or (record.get("cwd") if isinstance(record.get("cwd"), str) else None)
        if first is None and is_prompt(record) and not text_of(record).lstrip().startswith("<"):
            first = text_of(record)
        if session_id and cwd and first:
            break
    return session_id, cwd, first


def turn_status(last, mtime, now):
    if last.get("type") == "assistant":
        if last.get("isApiErrorMessage"):
            return "failed"
        if (last.get("message") or {}).get("stop_reason") in DONE_STOPS:
            return "completed"
    elif INTERRUPTED in text_of(last):
        return "interrupted"
    return "running" if now - mtime < RUNNING_WINDOW else "unknown"


def native_status(last):
    if last.get("type") == "assistant":
        return (last.get("message") or {}).get("stop_reason") or "streaming"
    return "interrupted" if INTERRUPTED in text_of(last) else "awaiting-model"


def summarize(head, tail, mtime, now):
    last, prompt, titles, last_prompt = tail_facts(tail)
    if last is None:
        return None  # metadata-only stub: the session was opened but never used
    session_id, cwd, first = head_facts(head)
    session_id = session_id or last.get("sessionId") or ""
    cwd = cwd or last.get("cwd") or ""
    if not isinstance(session_id, str) or not SESSION_ID.match(session_id) or not isinstance(cwd, str) or not cwd:
        return None
    title = next((titles[kind] for kind, _ in TITLES if kind in titles), None) or first or last_prompt
    status = turn_status(last, mtime, now)
    return {
        "sessionId": session_id, "cwd": cwd, "title": display_title(title),
        "status": status, "nativeStatus": native_status(last), "turnId": last.get("uuid"),
        "updatedAt": int(mtime), "startedAt": epoch(prompt.get("timestamp")) if prompt else None,
        "completedAt": epoch(last.get("timestamp")) if status == "completed" else None,
    }


def final_answer(blob):
    """The latest assistant text after the last typed prompt; never tool output or thinking."""
    for record in records(blob, reverse=True):
        if not is_main(record):
            continue
        if is_prompt(record):
            return ""
        if record.get("type") == "assistant" and text_of(record).strip():
            return text_of(record).strip()
    return ""


class ClaudeReader:
    agent, label = "claude", "Claude Code"

    def __init__(self, dirs=None):
        self.dirs = dirs
        self.cache = {}
        self.index = {}
        self.lock = threading.Lock()

    def sessions(self, now):
        dirs = self.dirs if self.dirs is not None else claude_dirs()
        if not dirs:
            raise Absent("没有发现 Claude Code 会话目录")
        found = {}
        cutoff = now - window_seconds()
        with self.lock:
            for config in dirs:
                for path in Path(config).glob("projects/*/*.jsonl"):
                    row = self.read(path, cutoff, now)
                    if row and (row["sessionId"] not in found or row["updatedAt"] > found[row["sessionId"]][0]["updatedAt"]):
                        found[row["sessionId"]] = (row, path, Path(config))
            self.index = {sid: (path, config, row["cwd"]) for sid, (row, path, config) in found.items()}
        return [dict(row, id="claude:" + sid) for sid, (row, _, _) in found.items()]

    def read(self, path, cutoff, now):
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_mtime < cutoff:
            return None
        key = (stat.st_size, stat.st_mtime_ns)
        cached = self.cache.get(path)
        # Unchanged files are not re-read; a running turn is re-checked because
        # it turns into unknown once the file stays quiet long enough.
        if cached and cached[0] == key and (cached[1] is None or cached[1]["status"] != "running"):
            return cached[1]
        try:
            head, tail = read_blobs(path, stat.st_size)
            row = summarize(head, tail, stat.st_mtime, now)
            if row is None and tail is not head:
                # The last record is longer than the tail window: read this one whole.
                whole = path.read_bytes()
                row = summarize(whole, whole, stat.st_mtime, now)
        except OSError:
            return None
        self.cache[path] = (key, row)
        return row

    def result(self, session_id):
        found = self.index.get(session_id)
        if not found:
            raise ValueError("会话不存在或不属于已识别的项目")
        path = found[0]
        try:
            head, tail = read_blobs(path, path.stat().st_size)
            text = final_answer(tail)
            if not text and tail is not head:
                text = final_answer(path.read_bytes())  # the answer is longer than the tail window
        except OSError as exc:
            raise Unavailable("Claude Code 会话文件暂不可读取：" + str(exc)) from exc
        if not text:
            return {"text": "本轮尚无最终答复，可以打开原对话查看进展。", "truncated": False}
        return {"text": text[:16000], "truncated": len(text) > 16000}

    def resume(self, session_id):
        found = self.index.get(session_id)
        if not found or not SESSION_ID.match(session_id):
            return None
        _, config, cwd = found
        env = ""
        if os.path.realpath(config) != os.path.realpath(default_claude_dir()):
            env = "CLAUDE_CONFIG_DIR=" + shlex.quote(str(config)) + " "
        return {"command": f"cd {shlex.quote(cwd)} && {env}claude --resume {session_id}", "runnable": True}


# ---------- magpie ----------

def magpie_binary():
    configured = os.environ.get("PULSE_MAGPIE_BIN")
    if configured is not None:
        return configured or None
    found = shutil.which("magpie")
    fallback = Path.home() / ".local" / "bin" / "magpie"
    return found or (str(fallback) if fallback.is_file() else None)


class MagpieReader:
    """Sessions magpie lists (`magpie sessions --json`) for agents not read directly here.

    Claude Code and Codex are read from their own files, which carry turn
    state; magpie adds the other agents it may read (Antigravity, OpenCode,
    Pi, Cursor…). The list takes seconds to build, so it is refreshed in the
    background and the dashboard never waits for it.
    """
    agent, label = "magpie", "magpie"
    SKIP = frozenset({"claude", "codex"})

    def __init__(self, binary=None, limit=300, ttl=120, background=True):
        self.binary = binary
        self.limit, self.ttl, self.background = limit, ttl, background
        self.rows, self.error, self.loaded_at = None, None, 0.0
        self.loading = False
        self.index = {}
        self.lock = threading.Lock()

    def sessions(self, now):
        binary = self.binary if self.binary is not None else magpie_binary()
        if not binary:
            raise Absent("没有发现 magpie")
        with self.lock:
            stale = time.time() - self.loaded_at > self.ttl
            start = stale and not self.loading
            if start:
                self.loading = True
        if start and self.background:
            threading.Thread(target=self.load, args=(binary,), daemon=True).start()
        elif start:
            self.load(binary)
        with self.lock:
            if self.rows is None and self.error:
                raise Unavailable("magpie 会话列表暂不可读取：" + self.error)
            return [row for row in (self.rows or []) if row["updatedAt"] >= now - window_seconds()]

    def load(self, binary):
        try:
            proc = subprocess.run([binary, "sessions", "--json", "--limit", str(self.limit)], capture_output=True,
                                  text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)
            if proc.returncode:
                raise Unavailable(f"magpie 退出码 {proc.returncode}")
            items = json.loads(proc.stdout)
            if not isinstance(items, list):
                raise Unavailable("magpie 返回的不是会话列表")
            rows, index = self.rows_of(items)
            with self.lock:
                self.rows, self.index, self.error = rows, index, None
        except (OSError, ValueError, subprocess.TimeoutExpired, Unavailable) as exc:
            with self.lock:
                self.error = str(exc) or type(exc).__name__
        finally:
            with self.lock:
                self.loaded_at, self.loading = time.time(), False

    def rows_of(self, items):
        rows, index = [], {}
        for item in items:
            if not isinstance(item, dict) or item.get("agent") in self.SKIP:
                continue
            agent, ident, cwd = str(item.get("agent") or ""), str(item.get("id") or ""), item.get("cwd")
            updated = epoch(item.get("last")) or epoch(item.get("start"))
            if not agent or not ident or not isinstance(cwd, str) or not cwd or updated is None:
                continue
            key = f"magpie:{agent}:{ident}"
            index[key] = item.get("resume") if isinstance(item.get("resume"), str) else ""
            rows.append({
                "id": key, "agent": agent, "agentLabel": AGENT_LABELS.get(agent, agent), "cwd": cwd,
                "title": display_title(item.get("title")), "status": "unknown", "nativeStatus": None,
                "turnId": None, "updatedAt": int(updated), "startedAt": epoch(item.get("start")), "completedAt": None,
            })
        return rows, index

    def resume(self, session_id):
        with self.lock:
            command = self.index.get(session_id)
        # magpie builds this from session data; show it for copying, never run it.
        return {"command": command, "runnable": False} if command else None


# ---------- all sources ----------

def temp_roots():
    """System temporary folders: scratch runs there are not projects (PULSE_SKIP_TEMP=0 keeps them)."""
    if os.environ.get("PULSE_SKIP_TEMP") == "0":
        return []
    roots = {tempfile.gettempdir(), "/tmp", "/var/tmp", "/var/folders"}
    return sorted({os.path.realpath(r) for r in roots} | roots)


def in_temp(cwd, roots):
    path = os.path.normpath(cwd)
    real = os.path.realpath(path)
    return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots for p in (path, real))


def folder_root(cwd, cache):
    """The repository a folder belongs to (nearest parent with .git), else the folder itself."""
    if cwd in cache:
        return cache[cwd]
    path = Path(os.path.normpath(cwd))
    root = path
    home = Path.home()
    for candidate in [path, *path.parents]:
        if candidate == home or candidate == candidate.parent:
            break
        if (candidate / ".git").exists():
            root = candidate
            break
    cache[cwd] = str(root)
    return cache[cwd]


def external_session(row, project_id, agent, label):
    return {
        "id": row["id"], "projectId": project_id, "title": row["title"], "status": row["status"],
        "nativeStatus": row["nativeStatus"], "turnId": row["turnId"], "updatedAt": row["updatedAt"],
        "completedAt": row["completedAt"], "startedAt": row["startedAt"],
        "unread": False, "unreadKnown": False, "readStateSource": "unavailable", "readRevision": None,
        "url": None, "agent": row.get("agent", agent), "agentLabel": row.get("agentLabel", label),
    }


LIMITATIONS = {
    "claude": "Claude Code 会话没有阅读状态，不计入未读；回合是否在运行只能从会话文件最近 30 分钟有无写入判断，更久没有动静显示状态未知。点击会话在终端里恢复对话。",
    "magpie": "magpie 列出的其他 agent 会话只有标题与时间，状态显示未知；点击会话复制恢复命令。",
}


class MultiReader:
    """Codex plus the other agents on this computer, merged into one project list."""

    def __init__(self, codex=None, others=None):
        names = enabled_sources()
        self.codex = codex if codex is not None else (CodexReader() if "codex" in names else None)
        if others is None:
            others = [ClaudeReader() if n == "claude" else MagpieReader() for n in names if n in ("claude", "magpie")]
        self.others = others
        self.roots = {}

    def snapshot(self):
        now = time.time()
        states = []
        base = self.codex_snapshot(now, states)
        projects, sessions = base["projects"], base["sessions"]
        for reader in self.others:
            try:
                rows = reader.sessions(now)
            except Absent:
                continue
            except Unavailable as exc:
                states.append({"id": reader.agent, "label": reader.label, "ok": False, "error": str(exc)})
                continue
            skip = temp_roots()
            rows = [row for row in rows if match_root(row["cwd"], projects) or not in_temp(row["cwd"], skip)]
            states.append({"id": reader.agent, "label": reader.label, "ok": True, "count": len(rows)})
            for row in rows:
                project_id = self.project_of(row["cwd"], projects)
                sessions.append(external_session(row, project_id, reader.agent, reader.label))
        if not any(s["ok"] for s in states):
            errors = [s["error"] for s in states if not s["ok"]]
            raise Unavailable("；".join(errors) or "本机没有发现 Codex、Claude Code 或 magpie 的会话")
        sessions.sort(key=lambda s: s["updatedAt"] or 0, reverse=True)
        return {"projects": projects, "sessions": sessions, "source": self.source(base["source"], states, now)}

    def codex_snapshot(self, now, states):
        empty = {"projects": [], "sessions": [], "source": None}
        if self.codex is None or not list(self.codex.home.glob("state_*.sqlite")):
            return empty
        try:
            base = self.codex.snapshot()
        except Unavailable as exc:
            states.append({"id": "codex", "label": "Codex", "ok": False, "error": str(exc)})
            return empty
        states.append({"id": "codex", "label": "Codex", "ok": True, "count": len(base["sessions"])})
        return base

    def project_of(self, cwd, projects):
        found = match_root(cwd, projects)
        if found:
            return found
        root = folder_root(cwd, self.roots)
        ident = "dir:" + hashlib.sha256(root.encode()).hexdigest()[:16]
        projects.append({"id": ident, "name": Path(root).name or root, "roots": [root]})
        return ident

    def source(self, codex_source, states, now):
        ok = [s for s in states if s["ok"]]
        source = dict(codex_source or {
            "mode": "local-readonly", "updatedAt": now, "refreshSeconds": 8, "compatibilityAdapter": True,
            "readStateAvailable": False, "readStateSource": "unavailable", "limitations": [],
        })
        source["label"] = "本机 " + " · ".join(s["label"] for s in ok)
        source["scope"] = "当前电脑 · Codex 已导入的本地项目与其他 agent 的工作目录 · 未归档主会话"
        source["sources"] = states
        source["limitations"] = list(source.get("limitations", [])) + [LIMITATIONS[s["id"]] for s in ok if s["id"] in LIMITATIONS]
        source["limitations"] += [f"{s['label']} 暂不可读取：{s['error']}" for s in states if not s["ok"]]
        return source

    def result(self, session_id):
        if session_id.startswith("claude:"):
            reader = next((r for r in self.others if r.agent == "claude"), None)
            if reader:
                return reader.result(session_id[len("claude:"):])
        if session_id.startswith("magpie:"):
            return {"text": "magpie 不提供这个会话的答复内容，可以打开原对话查看。", "truncated": False}
        if self.codex is None:
            raise ValueError("会话不存在或不属于已识别的项目")
        return self.codex.result(session_id)

    def resume(self, session_id):
        for reader in self.others:
            prefix = f"{reader.agent}:"
            if session_id.startswith(prefix):
                key = session_id[len(prefix):] if reader.agent == "claude" else session_id
                return reader.resume(key)
        return None


def terminal_command(command):
    """AppleScript that opens a Terminal window running `command` (macOS)."""
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    return ["osascript", "-e", 'tell application "Terminal"', "-e", "activate", "-e", f'do script "{escaped}"', "-e", "end tell"]
