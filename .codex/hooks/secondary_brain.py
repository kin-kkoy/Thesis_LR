"""Load compact project memory and validate memory maintenance for Codex.

The hook is deterministic and uses only the Python standard library. It never
writes semantic thesis memory. SessionStart injects a bounded digest and saves a
temporary baseline. Stop compares material changes with that baseline and asks
Codex to continue when a required memory update appears to be missing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MEMORY_FILES = {
    "docs/ai/PROJECT_STATE.md",
    "docs/ai/DECISIONS.md",
}
DIGEST_START = "<!-- BEGIN CODEX DIGEST -->"
DIGEST_END = "<!-- END CODEX DIGEST -->"
MAX_DIGEST_CHARACTERS = 7_000
GENERATED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "datafiles",
    "models",
    "output",
    "outputs",
    "processeddata",
    "checkpoints",
}
MATERIAL_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
NO_MEMORY_REASON = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?Memory impact\s*:\s*"
    r"(?:unchanged|none|not required)\s*(?:[-\u2014:]\s*)(.{12,})$"
)


class GitInspectionError(RuntimeError):
    """Raised when a read-only Git inspection cannot be completed."""


def emit(payload: dict[str, Any]) -> None:
    """Write the single JSON response expected by Codex hooks."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def read_event() -> dict[str, Any]:
    """Read a hook event from stdin, returning an empty event if malformed."""

    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def find_repository_root(cwd_value: object) -> Path | None:
    """Find the nearest parent containing .git and this hook script."""

    try:
        start = Path(str(cwd_value)).resolve() if cwd_value else Path.cwd().resolve()
    except OSError:
        start = Path.cwd().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (
            candidate / ".codex" / "hooks" / "secondary_brain.py"
        ).is_file():
            return candidate
    return None


def run_git(root: Path, *arguments: str) -> bytes:
    """Run a read-only Git query without changing global safe-directory settings."""

    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitInspectionError(detail or f"Git exited {completed.returncode}")
    return completed.stdout


def normalize_relative_path(value: str) -> str:
    """Normalize Git paths to repository-relative POSIX form."""

    normalized = PurePosixPath(value.replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def nul_paths(raw: bytes) -> set[str]:
    """Decode a NUL-delimited Git path list."""

    return {
        normalize_relative_path(item.decode("utf-8", errors="surrogateescape"))
        for item in raw.split(b"\0")
        if item
    }


def working_change_paths(root: Path) -> set[str]:
    """Return staged, unstaged, and untracked paths without ignored artifacts."""

    paths: set[str] = set()
    paths.update(nul_paths(run_git(root, "diff", "--name-only", "-z")))
    paths.update(nul_paths(run_git(root, "diff", "--cached", "--name-only", "-z")))
    paths.update(
        nul_paths(run_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    )
    return paths


def current_head(root: Path) -> str | None:
    """Return HEAD, allowing a repository with no commits."""

    try:
        return run_git(root, "rev-parse", "--verify", "HEAD").decode().strip() or None
    except GitInspectionError:
        return None


def is_memory_path(relative_path: str) -> bool:
    return normalize_relative_path(relative_path) in MEMORY_FILES


def is_material_path(relative_path: str) -> bool:
    """Classify source, configuration, and methodology paths for memory review."""

    normalized = normalize_relative_path(relative_path)
    lowered = normalized.casefold()
    parts = PurePosixPath(lowered).parts
    if normalized in MEMORY_FILES or any(
        part in GENERATED_DIRECTORY_NAMES for part in parts
    ):
        return False
    if lowered == "agents.md":
        return True
    if lowered.startswith(".codex/") or lowered.startswith(".github/agents/"):
        return True
    if lowered.startswith("docs/"):
        return Path(normalized).suffix.casefold() in MATERIAL_SUFFIXES
    if lowered in {"thesis_rf/agents.md", "thesis_lr/agents.md"}:
        return True
    suffix = Path(normalized).suffix.casefold()
    if lowered.startswith("thesis_rf/code/") or lowered.startswith("thesis_lr/"):
        return suffix in MATERIAL_SUFFIXES
    return suffix in {".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"}


def file_fingerprint(root: Path, relative_path: str) -> str:
    """Hash one source/memory file; represent deletion without failing."""

    path = root / PurePosixPath(relative_path)
    if not path.exists():
        return "<deleted>"
    if not path.is_file():
        return "<non-file>"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(128 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except OSError as error:
        return f"<unreadable:{type(error).__name__}>"


def working_fingerprints(root: Path) -> dict[str, str]:
    """Fingerprint only relevant dirty paths, never large ignored artifacts."""

    relevant = {
        path
        for path in working_change_paths(root)
        if is_material_path(path) or is_memory_path(path)
    }
    return {path: file_fingerprint(root, path) for path in sorted(relevant)}


def state_file(root: Path, session_id: object) -> Path:
    """Keep hook bookkeeping outside the repository."""

    repository_key = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:16]
    session_key = hashlib.sha256(str(session_id or "unknown").encode("utf-8")).hexdigest()
    return (
        Path(tempfile.gettempdir())
        / "codex-secondary-brain"
        / repository_key
        / f"{session_key}.json"
    )


def capture_baseline(root: Path, destination: Path) -> dict[str, Any]:
    """Create a session baseline once; compaction or resume must not replace it."""

    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    baseline = {
        "repository": str(root),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "head": current_head(root),
        "fingerprints": working_fingerprints(root),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return baseline


def read_baseline(destination: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def read_compact_digest(root: Path) -> str:
    """Return only the explicitly bounded startup section."""

    project_state = root / "docs" / "ai" / "PROJECT_STATE.md"
    try:
        text = project_state.read_text(encoding="utf-8")
    except OSError:
        return "Secondary Brain index is unavailable; inspect docs/ai/PROJECT_STATE.md."
    start = text.find(DIGEST_START)
    end = text.find(DIGEST_END, start + len(DIGEST_START)) if start >= 0 else -1
    if start < 0 or end < 0:
        return "Secondary Brain digest markers are missing; inspect PROJECT_STATE.md."
    digest = text[start + len(DIGEST_START) : end].strip()
    if len(digest) > MAX_DIGEST_CHARACTERS:
        digest = digest[:MAX_DIGEST_CHARACTERS].rstrip() + "\n[Digest truncated by hook.]"
    return digest


def committed_paths_since(root: Path, old_head: object, new_head: str | None) -> set[str]:
    """Include changes committed during the session, which final status omits."""

    if not isinstance(old_head, str) or not old_head or not new_head or old_head == new_head:
        return set()
    try:
        return nul_paths(run_git(root, "diff", "--name-only", "-z", old_head, new_head))
    except GitInspectionError:
        return set()


def changed_since_baseline(root: Path, baseline: dict[str, Any]) -> set[str]:
    """Compare working fingerprints and commits made in the session."""

    before_raw = baseline.get("fingerprints", {})
    before = before_raw if isinstance(before_raw, dict) else {}
    after = working_fingerprints(root)
    changed = {
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    }
    changed.update(committed_paths_since(root, baseline.get("head"), current_head(root)))
    return {normalize_relative_path(path) for path in changed}


def has_explicit_no_memory_reason(message: object) -> bool:
    return isinstance(message, str) and bool(NO_MEMORY_REASON.search(message))


def session_start(event: dict[str, Any], root: Path) -> None:
    warning: str | None = None
    try:
        capture_baseline(root, state_file(root, event.get("session_id")))
    except (GitInspectionError, OSError, ValueError) as error:
        warning = f"Secondary Brain baseline could not be recorded: {error}"
    head = current_head(root) or "unavailable"
    context = (
        "Secondary Brain startup context (navigation, not proof):\n"
        f"{read_compact_digest(root)}\n"
        f"Runtime Git HEAD: {head}. Inspect affected evidence when it differs from the reviewed state."
    )
    payload: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    if warning:
        payload["systemMessage"] = warning
    emit(payload)


def stop(event: dict[str, Any], root: Path) -> None:
    baseline = read_baseline(state_file(root, event.get("session_id")))
    if baseline is None:
        emit(
            {
                "systemMessage": (
                    "Secondary Brain completion check skipped because no session baseline "
                    "was available. Review memory impact manually."
                )
            }
        )
        return
    try:
        changed = changed_since_baseline(root, baseline)
    except (GitInspectionError, OSError, ValueError) as error:
        emit(
            {
                "systemMessage": (
                    "Secondary Brain completion check could not inspect the Git delta: "
                    f"{error}. Review memory impact manually."
                )
            }
        )
        return
    material = sorted(path for path in changed if is_material_path(path))
    memory_updates = sorted(path for path in changed if is_memory_path(path))
    if not material or memory_updates:
        emit({"continue": True})
        return
    if has_explicit_no_memory_reason(event.get("last_assistant_message")):
        emit({"continue": True})
        return
    preview = ", ".join(material[:8])
    if len(material) > 8:
        preview += f", and {len(material) - 8} more"
    reason = (
        "Material files changed during this task but neither docs/ai/PROJECT_STATE.md "
        "nor docs/ai/DECISIONS.md changed. Review the evidence, update the relevant "
        "memory once near completion, or—only if genuinely non-material—finish with "
        "`Memory impact: unchanged — <specific reason>.` Material paths: "
        f"{preview}."
    )
    if bool(event.get("stop_hook_active")):
        emit({"systemMessage": reason})
        return
    emit({"decision": "block", "reason": reason})


def main() -> None:
    event = read_event()
    root = find_repository_root(event.get("cwd"))
    event_name = event.get("hook_event_name")
    if root is None:
        if event_name == "Stop":
            emit({"continue": True})
        return
    if event_name == "SessionStart":
        session_start(event, root)
    elif event_name == "Stop":
        stop(event, root)


if __name__ == "__main__":
    main()
