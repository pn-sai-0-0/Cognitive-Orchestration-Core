"""
CognitiveOC v3 — Python Code Execution Tool
Static safety screen + subprocess isolation + timeout.
"""
from __future__ import annotations
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Patterns that will be blocked regardless of intent
_BLOCKED = re.compile(
    r'\b(import\s+(os|sys|subprocess|shutil|socket|ctypes|importlib|builtins)\b'
    r'|__import__|open\s*\(|exec\s*\(|compile\s*\()',
    re.I,
)


def execute(code: str, timeout: int = 5) -> dict:
    """Execute Python code in a sandboxed subprocess.

    Returns:
        {"ok": bool, "result": str, "stderr": str}
        or {"ok": False, "error": str}
    """
    if _BLOCKED.search(code):
        return {
            'ok':    False,
            'error': 'blocked by tool safety policy (filesystem/network/process access)',
        }

    with tempfile.NamedTemporaryFile(
        'w', suffix='.py', delete=False, prefix='coc_exec_'
    ) as f:
        f.write(code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, '-I', path],
            capture_output=True, text=True, timeout=timeout,
        )
        result = proc.stdout.strip()
        stderr = proc.stderr.strip()[-500:] if proc.stderr else ''
        return {
            'ok':     proc.returncode == 0,
            'result': result,
            'stderr': stderr,
        }
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'timed out after {timeout}s'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# Alias used by engine.py
run_python = execute
