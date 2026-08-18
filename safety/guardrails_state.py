"""
CognitiveOC v3 — Guardrail State Management
============================================

Hot-reload safe: reads var/guardrails_state.json on every call if mtime changed.
Profile switching applies a preset guard configuration.
Per-toggle: any cognitive guard can be set independently (mode becomes 'custom').
Hard integrity guards: NEVER stored here — always return True from IntegrityGuard.

File: safety/guardrails_state.py
Used by: safety/guardrails.py (is_enabled() calls), engine.py, ui/app.py
Persists: var/guardrails_state.json
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

try:
    from config import GUARDRAILS, GUARDRAIL_STATE, ensure_dirs
except ImportError:
    GUARDRAILS     = {}
    GUARDRAIL_STATE = Path("var/guardrails_state.json")
    def ensure_dirs(): pass

# ── Profile definitions ───────────────────────────────────────────────
_PROFILES: dict[str, dict] = {
    "strict": {
        "injection_check": True,  "jailbreak_detection": True,
        "pii_detection": True,    "pii_redaction": True,
        "output_filter": True,    "output_filtering": True,
        "tool_safety": True,      "retrieval_sanitization": True,
        "memory_safety": True,    "kg_validation": True,
        "workspace_validation": True, "judge_enforcement": True,
        "policy_enforcement": True,   "rate_limiting": True,
        "file_safety": True,
    },
    "standard": {
        "injection_check": True,  "jailbreak_detection": True,
        "pii_detection": True,    "pii_redaction": True,
        "output_filter": True,    "output_filtering": True,
        "tool_safety": True,      "retrieval_sanitization": True,
        "memory_safety": True,    "kg_validation": True,
        "workspace_validation": True, "judge_enforcement": False,
        "policy_enforcement": True,   "rate_limiting": True,
        "file_safety": True,
    },
    "research": {
        "injection_check": True,  "jailbreak_detection": False,
        "pii_detection": True,    "pii_redaction": False,
        "output_filter": False,   "output_filtering": False,
        "tool_safety": True,      "retrieval_sanitization": True,
        "memory_safety": True,    "kg_validation": False,
        "workspace_validation": False, "judge_enforcement": False,
        "policy_enforcement": False,  "rate_limiting": True,
        "file_safety": True,
    },
    "developer": {
        "injection_check": False, "jailbreak_detection": False,
        "pii_detection": False,   "pii_redaction": False,
        "output_filter": False,   "output_filtering": False,
        "tool_safety": False,     "retrieval_sanitization": False,
        "memory_safety": False,   "kg_validation": False,
        "workspace_validation": False, "judge_enforcement": False,
        "policy_enforcement": False,  "rate_limiting": False,
        "file_safety": True,      # always keep file safety on
    },
    "off": {
        k: False for k in [
            "injection_check", "jailbreak_detection", "pii_detection",
            "pii_redaction", "output_filter", "output_filtering",
            "tool_safety", "retrieval_sanitization", "memory_safety",
            "kg_validation", "workspace_validation", "judge_enforcement",
            "policy_enforcement",
        ]
    },
}
_PROFILES["off"].update({"rate_limiting": False, "file_safety": True})

# ── Defaults (standard profile) ───────────────────────────────────────
_DEFAULTS: dict[str, bool] = dict(_PROFILES["standard"])

# ── Module-level cache ────────────────────────────────────────────────
_state: dict | None = None
_state_mtime: float = 0.0
_state_lock  = threading.RLock()


def _path() -> Path:
    return Path(str(GUARDRAIL_STATE))


def _load() -> dict:
    """Load state from disk (hot-reload on mtime change)."""
    global _state, _state_mtime
    with _state_lock:
        p = _path()
        if p.exists():
            mtime = p.stat().st_mtime
            if _state is None or mtime > _state_mtime:
                try:
                    loaded       = json.loads(p.read_text())
                    _state       = {**_DEFAULTS, **loaded}
                    _state_mtime = mtime
                except Exception:
                    _state = dict(_DEFAULTS)
        else:
            _state = dict(_DEFAULTS)
        return _state


def _save():
    """Write current state to disk."""
    ensure_dirs()
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _state_lock:
        if _state is not None:
            payload = dict(_state)
            payload["_profile"]  = _state.get("_profile", "standard")
            payload["_updated"]  = time.strftime("%Y-%m-%dT%H:%M:%S")
            p.write_text(json.dumps(payload, indent=2))
            _state_mtime = p.stat().st_mtime


# ── Public API ────────────────────────────────────────────────────────

def is_enabled(name: str) -> bool:
    """Check if a cognitive guardrail is currently active.

    Hard integrity guards are NOT routed here — they always return True
    directly in IntegrityGuard. This function covers cognitive guards only.
    """
    return bool(_load().get(name, _DEFAULTS.get(name, True)))


def get() -> dict:
    """Return full state dict (copy), including profile and update timestamp."""
    state = dict(_load())
    state["_path"] = str(_path())
    return state


def set_guard(name: str, enabled: bool) -> dict:
    """Toggle a single cognitive guardrail.

    Switches profile to 'custom' so the change is visible in the UI.
    Audit logs the change.
    """
    _load()
    with _state_lock:
        if name not in _DEFAULTS:
            raise ValueError(
                f"Unknown guard '{name}'. Valid: {sorted(_DEFAULTS.keys())}"
            )
        _state[name]      = bool(enabled)
        _state["_profile"] = "custom"
    _save()
    _write_audit("guard_toggled", f"{name}={'on' if enabled else 'off'}")
    return get()


def set_profile(profile: str) -> dict:
    """Apply a complete guardrail profile.

    Valid profiles: strict | standard | research | developer | custom | off
    Audit logs the profile change.
    """
    valid = set(_PROFILES) | {"custom"}
    if profile not in valid:
        raise ValueError(f"Unknown profile '{profile}'. Valid: {sorted(valid)}")

    _load()
    with _state_lock:
        if profile == "custom":
            _state["_profile"] = "custom"
        else:
            preset = _PROFILES.get(profile, {})
            _state.update(preset)
            _state["_profile"] = profile
    _save()
    _write_audit("profile_applied", profile)
    return get()


def reset() -> dict:
    """Reset to standard profile defaults."""
    global _state
    with _state_lock:
        _state = {**_DEFAULTS, "_profile": "standard"}
    _save()
    _write_audit("reset_to_standard", "")
    return get()


def get_profiles() -> dict:
    """Return all available profile definitions."""
    return {k: dict(v) for k, v in _PROFILES.items()}


# ── Audit helper ──────────────────────────────────────────────────────
def _write_audit(event: str, detail: str):
    """Write to guardrail audit log (non-fatal)."""
    try:
        from config import LOG_DIR
        audit_path = Path(str(LOG_DIR)) / "guardrail_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps({
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event":  event,
            "detail": detail,
            "state":  {k: v for k, v in _load().items() if not k.startswith("_")},
        })
        with open(str(audit_path), "a") as f:
            f.write(record + "\n")
    except Exception:
        pass
