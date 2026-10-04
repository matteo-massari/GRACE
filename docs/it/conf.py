"""Italian documentation shares the English configuration and templates."""
from pathlib import Path

_parent = Path(__file__).resolve().parents[1] / "conf.py"
import runpy
_shared = runpy.run_path(str(_parent))
globals().update({key: value for key, value in _shared.items() if not key.startswith("_")})
language = "it"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_context = {"other_language_prefix": "../"}
