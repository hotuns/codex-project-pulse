"""Read-only Codex compatibility adapter and an independent personal task store.

Codex's SQLite schema is not a public API. Keep that dependency here, fail closed
on missing schemas, and never mutate or migrate a Codex-owned database.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path


class Unavailable(RuntimeError):
    pass


def display_title(raw):
    text = str(raw or '').strip()
    if '# Files mentioned by the user:' in text:
        marker = '## My request:'
        text = text.split(marker, 1)[1].strip() if marker in text else '含附件的会话'
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:150] + ('…' if len(text) > 150 else '') if text else '未命名会话'


def _open(uri):
    conn = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        # Opening is lazy; touch the schema so a failed open surfaces here.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def readonly(path: Path):
    if not path.is_file():
        raise Unavailable(f"找不到本机 Codex 数据文件：{path.name}")
    uri = path.resolve().as_uri()
    try:
        return _open(uri + "?mode=ro")
    except sqlite3.OperationalError:
        # A WAL-mode database whose -wal/-shm files were removed when the app
        # exited cannot be opened with mode=ro: SQLite would have to create the
        # shared-memory file. With no -wal content the main file is complete,
        # so reading it as immutable is exact and still never writes.
        wal = path.with_name(path.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise
        return _open(uri + "?immutable=1")


def status_of(turn, updated_at, now):
    if not turn:
        return "unknown"
    raw = turn["status"]
    if raw == "completed":
        return "completed"
    if raw in ("failed", "error"):
        return "failed"
    if raw in ("interrupted", "cancelled"):
        return "interrupted"
    if raw == "inProgress":
        # A persisted inProgress flag can survive a crashed app. Avoid claiming
        # those old turns are definitely still working.
        return "running" if now - updated_at < 7200 else "unknown"
    return "unknown"


def desktop_unread_index(state):
    """Resolve the desktop's account/host-scoped blue-dot state, read-only.

    The app's persisted v1 index was cross-checked against list_threads.isUnread.
    SQLite first_unread_turn can retain historical flags after the desktop has
    marked chats read, so it must never be used as a fallback here. A missing or
    ambiguous identity/host is unknown, not an instruction to merge accounts.
    """
    unknown = {"known": False, "ids": set(), "scope": None}
    data = state.get("electron-thread-read-state-v1")
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        return unknown
    identities = data.get("unreadByIdentity")
    migration = data.get("legacyMigration")
    if not isinstance(identities, dict) or len(identities) != 1 or not isinstance(migration, dict):
        return unknown
    identity, hosts = next(iter(identities.items()))
    adopted = migration.get("adoptedHostIds")
    if not isinstance(identity, str) or not isinstance(hosts, dict) or not isinstance(adopted, dict):
        return unknown
    host = adopted.get("local")
    if not isinstance(host, str) or not host.startswith("local:"):
        return unknown
    # Validate only the selected local host. Legacy lists and other hosts do not
    # describe the current desktop's unread state.
    ids = hosts.get(host)
    if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
        return unknown
    scope = hashlib.sha256((identity + "\n" + host).encode()).hexdigest()[:20]
    return {"known": True, "ids": set(ids), "scope": scope}


def match_root(cwd, projects):
    """The project whose root contains cwd; the longest root wins."""
    cwd = os.path.normpath(cwd)
    matches = [(len(root), p["id"]) for p in projects for root in p["roots"]
               if cwd == root or cwd.startswith(root.rstrip(os.sep) + os.sep)]
    return max(matches)[1] if matches else None


def project_for(row, projects, assignments, migration):
    explicit = assignments.get(row["id"], {}).get("projectId")
    by_id = {p["id"]: p for p in projects}
    if explicit in by_id:
        return explicit
    native = migration.get(row.get("project_id"), row.get("project_id"))
    if native in by_id:
        return native
    return match_root(row["cwd"], projects)


class CodexReader:
    def __init__(self, home=None):
        self.home = Path(home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")

    def state_path(self):
        paths = list(self.home.glob("state_*.sqlite"))
        if not paths:
            raise Unavailable("没有发现本机 Codex 会话数据库")
        return max(paths, key=lambda p: int(p.stem.split("_")[-1]))

    def snapshot(self):
        now = time.time()
        try:
            global_path = self.home / ".codex-global-state.json"
            state = json.loads(global_path.read_text()) if global_path.exists() else {}
            desktop_read = desktop_unread_index(state)
            names = {}
            index_path = self.home / 'session_index.jsonl'
            if index_path.exists():
                for line in index_path.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                        if entry.get('id') and entry.get('thread_name'):
                            names[entry['id']] = entry['thread_name']
                    except (ValueError, AttributeError):
                        continue  # An append may be partially written during polling.
            with closing(readonly(self.state_path())) as db:
                columns = {r[1] for r in db.execute("PRAGMA table_info(threads)")}
                if not {"id", "cwd", "title", "updated_at", "archived"} <= columns:
                    raise Unavailable("Codex 数据结构已变化，需要更新兼容适配器")
                projects, migration = self._projects(db, state)
                fields = [k for k in ("id", "cwd", "title", "name", "updated_at", "updated_at_ms", "project_id", "source", "agent_path") if k in columns]
                sql = "SELECT " + ",".join(fields) + " FROM threads WHERE archived=0"
                if "agent_path" in columns:
                    sql += " AND (agent_path IS NULL OR agent_path='/root')"
                rows = [dict(r) for r in db.execute(sql + " ORDER BY updated_at DESC")]
            sessions = []
            with closing(readonly(self.home / "thread_history_1.sqlite")) as history:
                for row in rows:
                    # Older versions encoded subagents in source, without agent_path.
                    if 'subagent' in str(row.get("source", "")).lower():
                        continue
                    project_id = project_for(row, projects, state.get("thread-project-assignments", {}), migration)
                    if not project_id:
                        continue
                    turn = history.execute("SELECT turn_id,status,started_at,completed_at FROM thread_turns WHERE thread_id=? ORDER BY rollout_ordinal DESC LIMIT 1", (row["id"],)).fetchone()
                    status = status_of(turn, row["updated_at"], now)
                    unread = row["id"] in desktop_read["ids"]
                    turn_id = turn["turn_id"] if turn else None
                    sessions.append({
                        "id": row["id"], "projectId": project_id,
                        "title": display_title(row.get("name") or names.get(row['id']) or row["title"]),
                        "status": status, "nativeStatus": turn["status"] if turn else None,
                        "turnId": turn["turn_id"] if turn else None,
                        "updatedAt": row["updated_at"],
                        "completedAt": turn["completed_at"] if turn else None,
                        "startedAt": turn["started_at"] if turn else None,
                        "unread": unread,
                        "unreadKnown": desktop_read["known"],
                        "readStateSource": "desktop-v1" if desktop_read["known"] else "unavailable",
                        "readRevision": f"desktop-v1:{desktop_read['scope']}:{turn_id}:{int(unread)}" if desktop_read["known"] else None,
                        "url": "codex://threads/" + row["id"],
                        "agent": "codex", "agentLabel": "Codex",
                    })
            return {"projects": projects, "sessions": sessions, "source": {
                "mode": "local-readonly", "label": "本机 Codex", "updatedAt": now,
                "refreshSeconds": 8, "compatibilityAdapter": True,
                "readStateAvailable": desktop_read["known"],
                "readStateSource": "desktop-v1" if desktop_read["known"] else "unavailable",
                "scope": "当前电脑 · 已导入的本地项目 · 未归档主会话",
                "limitations": ["通过本机只读适配器接入；Codex 升级后可能需要适配。", "未读跟随桌面阅读状态；无法确定账号或主机时显示未知，不使用历史数据库标记推断。", "未接入云端与远程主机。运行中状态暂不能细分为等待审批或等待输入。", "长时间没有活动的运行记录显示为状态未知，不将会话结束等同于项目完成。"],
            }}
        except (sqlite3.Error, ValueError, OSError, KeyError) as exc:
            raise Unavailable("本机 Codex 数据暂不可读取：" + str(exc)) from exc

    def _projects(self, db, state):
        local = state.get("local-projects", {})
        migration = {}
        for host, mapping in state.get("app-server-project-id-by-legacy-project-id-by-host", {}).items():
            if host == f"local:{self.home}":
                migration.update({native: legacy for legacy, native in mapping.items()})
        projects = [{"id": key, "name": p.get("name", key), "roots": [os.path.normpath(r) for r in p.get("rootPaths", [])]} for key, p in local.items()]
        if not projects:
            try:
                projects = [{"id": r["id"], "name": r["name"], "roots": [v[0] for v in db.execute("SELECT path FROM project_roots WHERE project_id=? ORDER BY position", (r["id"],))]} for r in db.execute("SELECT id,name FROM projects ORDER BY position")]
            except sqlite3.Error as exc:
                raise Unavailable("当前 Codex 版本没有可识别的项目目录") from exc
        return projects, migration

    def result(self, thread_id):
        # Return only final user-visible output, never tool output or reasoning.
        with closing(readonly(self.home / "thread_history_1.sqlite")) as db:
            turn = db.execute("SELECT turn_id,status FROM thread_turns WHERE thread_id=? ORDER BY rollout_ordinal DESC LIMIT 1", (thread_id,)).fetchone()
            if not turn:
                return {"text": "暂无可读取的本轮结果。", "truncated": False}
            rows = db.execute("SELECT item_json FROM thread_items WHERE thread_id=? AND turn_id=? AND item_type='agentMessage' ORDER BY rollout_ordinal DESC LIMIT 20", (thread_id, turn["turn_id"]))
            for row in rows:
                item = json.loads(row[0])
                if item.get("phase") == "final_answer":
                    text = item.get("text", "")
                    return {"text": text[:16000], "truncated": len(text) > 16000}
            return {"text": "本轮尚无最终答复，可以跳转到原对话查看进展。", "truncated": False}


class Store:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get("PULSE_DATA_DIR") or Path.home() / ".local/share/codex-project-pulse")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "pulse.sqlite3"
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS todos (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                  project_id TEXT NOT NULL, session_id TEXT, priority TEXT NOT NULL,
                  due_date TEXT, status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS acknowledgements (session_id TEXT PRIMARY KEY, turn_id TEXT, read_revision TEXT);
                CREATE TABLE IF NOT EXISTS project_preferences (project_id TEXT PRIMARY KEY, pinned INTEGER NOT NULL DEFAULT 0);
            """)
            db.commit()
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def enrich(self, snapshot):
        with closing(self.connect()) as db:
            todos = [self.todo_row(r) for r in db.execute("SELECT * FROM todos ORDER BY created_at DESC")]
            ack = {r["session_id"]: r for r in db.execute("SELECT * FROM acknowledgements")}
            pinned = {r[0] for r in db.execute("SELECT project_id FROM project_preferences WHERE pinned=1")}
        for s in snapshot["sessions"]:
            a = ack.get(s["id"])
            s["acknowledged"] = bool(a and a["turn_id"] == s["turnId"] and a["read_revision"] == s["readRevision"])
            # Processing a result and reading it in Codex are separate states.
            s["needsReview"] = s["status"] == "completed" and s["unreadKnown"] and s["unread"]
        for p in snapshot["projects"]:
            items = [s for s in snapshot["sessions"] if s["projectId"] == p["id"]]
            p.update({"pinned": p["id"] in pinned, "sessionCount": len(items), "running": sum(s["status"] == "running" for s in items), "unread": sum(s["needsReview"] for s in items), "failed": sum(s["status"] == "failed" for s in items), "todoCount": sum(t["projectId"] == p["id"] and t["status"] != "done" for t in todos), "updatedAt": max((s["updatedAt"] for s in items), default=0)})
            # Only Codex has a read state. Other agents' sessions neither count as
            # unread nor make the whole project's unread count unknown.
            p["unreadKnown"] = all(s["unreadKnown"] for s in items if s.get("agent", "codex") == "codex")
        snapshot["todos"] = todos
        # The entrypoint badge follows unread chats regardless of their current
        # turn status. The existing completed-unread inbox keeps its own count.
        known = snapshot.get("source", {}).get("readStateAvailable") is True
        snapshot["unreadBadge"] = {"known": known, "count":
            sum(s["unreadKnown"] and s["unread"] for s in snapshot["sessions"]) if known else None}
        return snapshot

    @staticmethod
    def todo_row(r):
        return {"id": r["id"], "title": r["title"], "projectId": r["project_id"], "sessionId": r["session_id"], "priority": r["priority"], "dueDate": r["due_date"], "status": r["status"], "createdAt": r["created_at"], "updatedAt": r["updated_at"]}

    def save_todo(self, values, snapshot, todo_id=None):
        old = None
        with closing(self.connect()) as db:
            if todo_id:
                r = db.execute("SELECT * FROM todos WHERE id=?", (todo_id,)).fetchone()
                if not r:
                    raise ValueError("待办不存在")
                old = self.todo_row(r)
            v = {**(old or {}), **values}
            title = str(v.get("title", "")).strip()
            if not title or len(title) > 240:
                raise ValueError("待办标题需要 1–240 个字符")
            retained_project = old and old['projectId'] == v.get('projectId')
            if not retained_project and v.get("projectId") not in {p["id"] for p in snapshot["projects"]}:
                raise ValueError("请选择已导入的项目")
            session = v.get("sessionId") or None
            retained_session = old and retained_project and old['sessionId'] == session
            if session and not retained_session and not any(s["id"] == session and s["projectId"] == v["projectId"] for s in snapshot["sessions"]):
                raise ValueError("绑定的会话必须属于所选项目")
            priority, status = v.get("priority", "normal"), v.get("status", "todo")
            if priority not in ("high", "normal", "low") or status not in ("todo", "doing", "done"):
                raise ValueError("待办优先级或状态无效")
            due = v.get("dueDate") or None
            if due:
                from datetime import date
                try:
                    if date.fromisoformat(due).isoformat() != due:
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError("截止日期格式应为 YYYY-MM-DD")
            ident, now = todo_id or str(uuid.uuid4()), time.time()
            db.execute("INSERT OR REPLACE INTO todos VALUES (?,?,?,?,?,?,?,?,?)", (ident, title, v["projectId"], session, priority, due, status, old["createdAt"] if old else now, now))
            db.commit()
            return self.todo_row(db.execute("SELECT * FROM todos WHERE id=?", (ident,)).fetchone())

    def delete_todo(self, todo_id):
        with closing(self.connect()) as db:
            if db.execute("DELETE FROM todos WHERE id=?", (todo_id,)).rowcount == 0:
                raise ValueError("待办不存在")
            db.commit()

    def acknowledge(self, session):
        with closing(self.connect()) as db:
            db.execute("INSERT OR REPLACE INTO acknowledgements VALUES (?,?,?)", (session["id"], session["turnId"], session["readRevision"]))
            db.commit()

    def pin(self, project_id, pinned):
        with closing(self.connect()) as db:
            db.execute("INSERT OR REPLACE INTO project_preferences VALUES (?,?)", (project_id, int(bool(pinned))))
            db.commit()
