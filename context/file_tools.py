"""
File tools for Atlas Realty agent system.
Gives Dev Agents the ability to read and write actual repo files.
Parses LLM markdown output and applies code blocks to the real codebase.
"""

from __future__ import annotations
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

# agent-system/ → atlas-realty/ (one level up)
REPO_ROOT = Path(__file__).parent.parent.parent

# Extensions we allow agents to write
ALLOWED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".sql", ".json", ".css", ".md", ".env",
    ".sh", ".txt", ".yml", ".yaml",
}


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> list[dict]:
    """
    Parse LLM output and extract (path, language, content) triples.

    Handles the pattern that dev agent prompts instruct Claude to follow:

        ### `path/to/file.py`
        ```python
        [full file content]
        ```

    Also handles without backticks around path:

        ### path/to/file.py
        ```python
        [content]
        ```
    """
    results = []

    # Primary pattern: ### `path` or ### path, then a code fence
    pattern = re.compile(
        r'###\s+`?([^`\n]+\.\w+)`?\s*\n'   # ### `path/to/file.ext`
        r'```(\w*)\n'                         # ```python
        r'(.*?)'                              # content (lazy)
        r'```',                               # closing fence
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        raw_path = match.group(1).strip()
        language = match.group(2).strip()
        content = match.group(3).rstrip("\n")

        # Normalise path: strip leading slashes and whitespace
        path_str = raw_path.lstrip("/ \t")

        # Only allow known extensions
        suffix = Path(path_str).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            continue

        results.append({
            "path": path_str,
            "language": language,
            "content": content,
        })

    return results


# ---------------------------------------------------------------------------
# Repo file writer
# ---------------------------------------------------------------------------

def write_repo_files(blocks: list[dict]) -> list[str]:
    """
    Write code blocks to actual repo files.
    Creates parent directories if they don't exist.
    Returns list of relative paths that were written.
    Skips any path that tries to escape the repo root (safety check).
    """
    written: list[str] = []

    for block in blocks:
        path_str = block["path"]
        content = block["content"]

        full_path = (REPO_ROOT / path_str).resolve()

        # Safety: must be inside the repo
        try:
            full_path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            print(f"[file_tools] SKIPPED (path escape attempt): {path_str}")
            continue

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content + "\n", encoding="utf-8")
        written.append(path_str)
        print(f"[file_tools] WROTE: {path_str}")

    return written


# ---------------------------------------------------------------------------
# Repo file reader
# ---------------------------------------------------------------------------

def read_repo_file(path_str: str) -> str:
    """Read a file from the repo by relative path. Returns content or error string."""
    full_path = (REPO_ROOT / path_str.lstrip("/")).resolve()
    if not full_path.exists():
        return f"(file not found: {path_str})"
    try:
        return full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"(error reading {path_str}: {e})"


def read_repo_files(paths: list[str]) -> dict[str, str]:
    """Read multiple repo files. Returns {relative_path: content}."""
    return {p: read_repo_file(p) for p in paths}


# ---------------------------------------------------------------------------
# Arch-spec file list extractor
# ---------------------------------------------------------------------------

def extract_file_paths_from_arch_spec(arch_spec: str) -> list[str]:
    """
    Pull the list of files to create/modify from arch-spec.md.
    Looks in '## Files to Create' and '## Files to Modify' sections.
    Returns relative paths like ['backend/main.py', 'frontend/types/listing.ts'].
    """
    paths: list[str] = []
    # Match: - path/to/file.ext  or  - `path/to/file.ext`
    line_pattern = re.compile(
        r'^\s*-\s+`?([^\s`]+\.(?:py|ts|tsx|js|jsx|sql|json|css|md))`?'
    )
    in_section = False

    for line in arch_spec.split("\n"):
        if re.match(r'^##\s+(Files to Create|Files to Modify|New Components)', line):
            in_section = True
        elif re.match(r'^##\s+', line) and in_section:
            in_section = False

        if in_section:
            m = line_pattern.match(line)
            if m:
                paths.append(m.group(1))

    return paths


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def docker_is_running() -> bool:
    """Return True if docker compose services are reachable."""
    result = subprocess.run(
        ["docker", "compose", "ps", "--quiet"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode == 0


def docker_rebuild(services: list[str], timeout: int = 300) -> dict:
    """
    Run `docker compose up --build -d <services>`.
    Returns {"success": bool, "stdout": str, "stderr": str}.
    """
    cmd = ["docker", "compose", "up", "--build", "-d"] + services
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(cmd),
    }


def docker_exec(service: str, command: list[str], timeout: int = 120) -> dict:
    """
    Run a command inside a running container via `docker compose exec -T`.
    Returns {"success": bool, "output": str, "returncode": int}.
    """
    cmd = ["docker", "compose", "exec", "-T", service] + command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )
    combined = result.stdout + result.stderr
    return {
        "success": result.returncode == 0,
        "output": combined,
        "returncode": result.returncode,
    }


def wait_for_healthy(service: str, retries: int = 12, delay: int = 5) -> bool:
    """Poll until a service reports healthy or retries exhausted."""
    for _ in range(retries):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", f"atlas_{service}"],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip()
        if status == "healthy":
            return True
        if status not in ("starting", ""):
            return False
        time.sleep(delay)
    return False
