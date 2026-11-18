import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"
sys.path.insert(0, str(SOURCE))
from project_pulse import __version__
from project_pulse.agents import ClaudeReader, MagpieReader, MultiReader, epoch, terminal_command
from project_pulse.data import CodexReader, Store, Unavailable
from project_pulse.server import Service

ISO = "2026-11-17T10:00:00.000Z"


def record(kind, sid, cwd, content=None, **extra):
    base = {"type": kind, "sessionId": sid, "cwd": cwd, "uuid": str(uuid.uuid4()), "timestamp": ISO}
    if content is not None:
        base["message"] = {"role": kind, "content": content, **({"stop_reason": extra.pop("stop")} if "stop" in extra else {})}
    base.update(extra)
    return base


def prompt(sid, cwd, text):
    return record("user", sid, cwd, text)


def answer(sid, cwd, text, stop="end_turn", **extra):
    return record("assistant", sid, cwd, [{"type": "text", "text": text}], stop=stop, **extra)


def tool_call(sid, cwd):
    return record("assistant", sid, cwd, [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], stop="tool_use")


class AgentsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Never read the developer's real agents from tests.
        self.env = mock.patch.dict(os.environ, {"PULSE_CLAUDE_DIRS": "", "PULSE_MAGPIE_BIN": "", "PULSE_DISABLE_OPEN": "1",
                                                    # Fixtures live in the temp folder, which is skipped by default.
                                                    "PULSE_SKIP_TEMP": "0"})
        self.env.start()
        self.codex_home = self.root / "codex"
        self.codex_home.mkdir()
        self.claude = self.root / "claude-config"
        (self.claude / "projects").mkdir(parents=True)
        self.repo = self.root / "repo"
        (self.repo / ".git").mkdir(parents=True)
        (self.repo / "pkg").mkdir()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def transcript(self, cwd, records, mtime=None, sid=None, config=None):
        sid = sid or str(uuid.uuid4())
        folder = (config or self.claude) / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sid}.jsonl"
        with path.open("w") as fh:
            for item in records(sid) if callable(records) else records:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return sid

    def reader(self, codex=None, others=None):
        return MultiReader(codex=codex or CodexReader(self.codex_home), others=others if others is not None else [ClaudeReader([self.claude])])

    def test_claude_sessions_report_turn_state_and_group_by_repository(self):
        cwd = str(self.repo / "pkg")
        now = time.time()
        done = self.transcript(cwd, lambda s: [
            {"type": "permission-mode", "sessionId": s}, prompt(s, cwd, "修一下导出"),
            record("assistant", s, cwd, [{"type": "thinking", "thinking": "never shown"}]),
            answer(s, cwd, "Sidechain text", isSidechain=True), answer(s, cwd, "导出已修好"),
            {"type": "ai-title", "sessionId": s, "aiTitle": "修复导出"}])
        running = self.transcript(cwd, lambda s: [prompt(s, cwd, "跑测试"), tool_call(s, cwd)])
        stale = self.transcript(cwd, lambda s: [prompt(s, cwd, "旧任务"), tool_call(s, cwd)], mtime=now - 4000)
        stopped = self.transcript(cwd, lambda s: [prompt(s, cwd, "停下"), tool_call(s, cwd), prompt(s, cwd, "[Request interrupted by user]")])
        failed = self.transcript(cwd, lambda s: [prompt(s, cwd, "调用失败"), answer(s, cwd, "API Error", stop=None, isApiErrorMessage=True)])
        self.transcript(cwd, lambda s: [{"type": "mode", "sessionId": s, "mode": "default"}, {"type": "last-prompt", "sessionId": s, "lastPrompt": "x"}])
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.claude.rglob("*.jsonl")}
        snap = self.reader().snapshot()
        by_id = {s["id"]: s for s in snap["sessions"]}
        self.assertEqual(set(by_id), {f"claude:{s}" for s in (done, running, stale, stopped, failed)})
        self.assertEqual({k: v["status"] for k, v in by_id.items()}, {
            f"claude:{done}": "completed", f"claude:{running}": "running", f"claude:{stale}": "unknown",
            f"claude:{stopped}": "interrupted", f"claude:{failed}": "failed"})
        self.assertEqual(by_id[f"claude:{done}"]["title"], "修复导出")
        self.assertEqual(by_id[f"claude:{stale}"]["title"], "旧任务")
        self.assertEqual({s["agent"] for s in snap["sessions"]}, {"claude"})
        self.assertFalse(any(s["unreadKnown"] or s["unread"] for s in snap["sessions"]))
        # A folder inside a git repository is grouped under the repository.
        self.assertEqual([(p["name"], p["roots"]) for p in snap["projects"]], [("repo", [str(self.repo)])])
        self.assertEqual({s["projectId"] for s in snap["sessions"]}, {snap["projects"][0]["id"]})
        reader = self.reader()
        reader.snapshot()
        self.assertEqual(reader.result(f"claude:{done}")["text"], "导出已修好")
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.claude.rglob("*.jsonl")})

    def test_claude_sessions_join_codex_projects_and_keep_project_unread_known(self):
        (self.codex_home / ".codex-global-state.json").write_text(json.dumps({"local-projects": {"p1": {"name": "Repo", "rootPaths": [str(self.repo)]}}}))
        import sqlite3
        with sqlite3.connect(self.codex_home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads(id TEXT, cwd TEXT, title TEXT, updated_at INTEGER, archived INTEGER)")
        with sqlite3.connect(self.codex_home / "thread_history_1.sqlite") as db:
            db.execute("CREATE TABLE thread_turns(thread_id TEXT,turn_id TEXT,status TEXT,started_at INTEGER,completed_at INTEGER,rollout_ordinal INTEGER)")
        cwd = str(self.repo / "pkg")
        sid = self.transcript(cwd, lambda s: [prompt(s, cwd, "看看"), answer(s, cwd, "好了")])
        service = Service(self.reader(), Store(self.root / "personal"))
        snap = service.snapshot(force=True)
        self.assertEqual([p["id"] for p in snap["projects"]], ["p1"])
        self.assertEqual(snap["sessions"][0]["projectId"], "p1")
        self.assertEqual([s["id"] for s in snap["source"]["sources"] if s["ok"]], ["codex", "claude"])
        # Claude has no read state; it must not make the project's unread count unknown.
        self.assertIs(snap["projects"][0]["unreadKnown"], True)
        todo = service.action("todo", {"title": "跟进", "projectId": "p1", "sessionId": f"claude:{sid}"})
        self.assertEqual(todo["sessionId"], f"claude:{sid}")

    def test_a_long_final_record_beyond_the_tail_window_is_still_read(self):
        cwd = str(self.repo)
        filler = [{"type": "api-request", "sessionId": "x", "body": "z" * 4000}] * 120
        sid = self.transcript(cwd, lambda s: [prompt(s, cwd, "写长文")] + filler + [answer(s, cwd, "长" * 400000)])
        reader = self.reader()
        snap = reader.snapshot()
        self.assertEqual(snap["sessions"][0]["status"], "completed")
        self.assertEqual(snap["sessions"][0]["title"], "写长文")
        result = reader.result(f"claude:{sid}")
        self.assertTrue(result["truncated"])

    def test_one_missing_agent_does_not_hide_the_others(self):
        cwd = str(self.repo)
        self.transcript(cwd, lambda s: [prompt(s, cwd, "只有 Claude"), answer(s, cwd, "ok")])
        snap = self.reader().snapshot()  # no Codex database at all
        self.assertEqual(len(snap["sessions"]), 1)
        self.assertEqual(snap["source"]["readStateAvailable"], False)
        self.assertIn("Claude Code", snap["source"]["label"])
        with self.assertRaisesRegex(Unavailable, "没有发现"):
            self.reader(others=[ClaudeReader([])]).snapshot()

    def test_claude_session_resumes_in_its_own_config_folder(self):
        cwd = str(self.root / "it's here")
        sid = self.transcript(cwd, lambda s: [prompt(s, cwd, "hi"), answer(s, cwd, "hello")])
        service = Service(self.reader(), Store(self.root / "personal"))
        service.snapshot(force=True)
        opened = service.action("open", {"id": f"claude:{sid}"})
        self.assertFalse(opened["dispatched"])
        self.assertEqual(opened["command"], f"cd '{self.root}/it'\"'\"'s here' && CLAUDE_CONFIG_DIR={self.claude} claude --resume {sid}")
        script = terminal_command('echo "a\\b"')
        self.assertEqual(script[script.index("-e", 4) + 1], 'do script "echo \\"a\\\\b\\""')

    def test_magpie_adds_other_agents_and_its_commands_are_only_copied(self):
        items = [
            {"agent": "claude", "id": "c1", "cwd": str(self.repo), "title": "dup", "last": "2026-11-17T10:00:00.07Z"},
            {"agent": "codex", "id": "x1", "cwd": str(self.repo), "title": "dup", "last": ISO},
            {"agent": "antigravity", "id": "a1", "cwd": str(self.repo / "pkg"), "title": "Antigravity chat",
             "start": "2026-11-17T09:00:00.669519958-08:00", "last": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "resume": "cd '/x' && touch /tmp/should-not-run"},
        ]
        fake = self.root / "bin" / "magpie"
        fake.parent.mkdir()
        fake.write_text(f"#!{sys.executable}\nimport json, sys\nassert sys.argv[1:3] == ['sessions', '--json']\nprint(json.dumps({json.dumps(items)}))\n")
        fake.chmod(0o755)
        magpie = MagpieReader(binary=str(fake), background=False)
        service = Service(self.reader(others=[magpie]), Store(self.root / "personal"))
        snap = service.snapshot(force=True)
        self.assertEqual([(s["id"], s["agentLabel"], s["status"]) for s in snap["sessions"]], [("magpie:antigravity:a1", "Antigravity", "unknown")])
        self.assertEqual(snap["projects"][0]["roots"], [str(self.repo)])
        os.environ.pop("PULSE_DISABLE_OPEN")
        opened = service.action("open", {"id": "magpie:antigravity:a1"})
        self.assertEqual(opened, {"ok": True, "dispatched": False, "command": items[2]["resume"]})

    def test_magpie_failure_is_reported_without_hiding_claude(self):
        cwd = str(self.repo)
        self.transcript(cwd, lambda s: [prompt(s, cwd, "hi"), answer(s, cwd, "ok")])
        broken = MagpieReader(binary=str(self.root / "missing-magpie"), background=False)
        snap = self.reader(others=[ClaudeReader([self.claude]), broken]).snapshot()
        self.assertEqual(len(snap["sessions"]), 1)
        self.assertEqual([(s["id"], s["ok"]) for s in snap["source"]["sources"]], [("claude", True), ("magpie", False)])

    def test_scratch_runs_in_temp_folders_are_not_projects(self):
        cwd = str(self.repo)
        self.transcript(cwd, lambda s: [prompt(s, cwd, "临时探测"), answer(s, cwd, "ok")])
        os.environ.pop("PULSE_SKIP_TEMP")
        snap = self.reader().snapshot()
        self.assertEqual((snap["sessions"], snap["projects"]), ([], []))
        (self.codex_home / ".codex-global-state.json").write_text(json.dumps({"local-projects": {"p1": {"name": "Repo", "rootPaths": [cwd]}}}))
        import sqlite3
        with sqlite3.connect(self.codex_home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads(id TEXT, cwd TEXT, title TEXT, updated_at INTEGER, archived INTEGER)")
        with sqlite3.connect(self.codex_home / "thread_history_1.sqlite") as db:
            db.execute("CREATE TABLE thread_turns(thread_id TEXT,turn_id TEXT,status TEXT,started_at INTEGER,completed_at INTEGER,rollout_ordinal INTEGER)")
        # A folder the user imported into Codex counts even when it is under a temp folder.
        self.assertEqual(len(self.reader().snapshot()["sessions"]), 1)

    def test_iso_timestamps_from_both_agents(self):
        self.assertAlmostEqual(epoch("2026-11-18T03:57:41.07Z"), epoch("2026-11-18T03:57:41.070000+00:00"))
        self.assertAlmostEqual(epoch("2026-11-13T05:52:49.669519958-08:00"), epoch("2026-11-13T13:52:49.669519Z"))
        self.assertIsNone(epoch("not a date"))

    def test_stdio_launcher_runs_from_any_folder_for_claude_code_and_magpie(self):
        installed = self.root / "installed plugin"
        shutil.copytree(SOURCE, installed / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "assets", installed / "assets")
        cwd = str(self.repo)
        sid = self.transcript(cwd, lambda s: [prompt(s, cwd, "from launcher"), answer(s, cwd, "ok")])
        manifest = json.loads((REPO / ".claude-plugin/plugin.json").read_text())["mcpServers"]["project_pulse"]
        args = [a.replace("${CLAUDE_PLUGIN_ROOT}", str(installed)) for a in manifest["args"]]
        env = {**os.environ, "CODEX_HOME": str(self.codex_home), "PULSE_DATA_DIR": str(self.root / "personal"), "PULSE_CLAUDE_DIRS": str(self.claude)}
        env.pop("PYTHONPATH", None)
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_sessions", "arguments": {}}}]
        proc = subprocess.run([sys.executable, *args], cwd=self.root, env=env, text=True, capture_output=True, timeout=20,
                              input="\n".join(json.dumps(m) for m in messages) + "\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        replies = {r["id"]: r for r in map(json.loads, proc.stdout.splitlines())}
        sessions = json.loads(replies[2]["result"]["content"][0]["text"])["sessions"]
        self.assertEqual([(s["id"], s["agent"]) for s in sessions], [(f"claude:{sid}", "claude")])

    def test_every_manifest_and_the_ui_bridge_carry_one_version(self):
        versions = {name: json.loads((REPO / name).read_text())["version"] for name in ("plugin.json", ".codex-plugin/plugin.json", ".claude-plugin/plugin.json")}
        self.assertEqual(set(versions.values()), {__version__}, versions)
        self.assertIn(f"version: '{__version__}'", (SOURCE / "project_pulse/web/mcp-bridge.js").read_text())
        marketplace = json.loads((REPO / ".claude-plugin/marketplace.json").read_text())
        self.assertEqual(marketplace["plugins"][0]["name"], json.loads((REPO / ".claude-plugin/plugin.json").read_text())["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
