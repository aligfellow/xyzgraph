"""Configuration loading for xyzrender."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from xyzrender.types import RenderConfig, resolve_color

logger = logging.getLogger(__name__)

_PRESET_DIR = Path(__file__).parent / "presets"


def _load_default() -> dict:
    """Return the built-in default preset."""
    return json.loads((_PRESET_DIR / "default.json").read_text())


def load_config(name_or_path: str) -> dict:
    """Load config from a built-in preset name or a JSON file path.

    Built-in presets are loaded directly.  User-provided JSON files are
    merged on top of the ``default`` preset so that any unspecified keys
    inherit the standard defaults.
    """
    # Built-in preset?
    preset_file = _PRESET_DIR / f"{name_or_path}.json"
    if preset_file.exists():
        logger.debug("Loading preset: %s", preset_file)
        return json.loads(preset_file.read_text())

    # User-provided file path — layer on top of default preset
    path = Path(name_or_path)
    if path.exists():
        logger.debug("Loading config file: %s (on top of default)", path)
        base = _load_default()
        user = json.loads(path.read_text())
        # Deep-merge nested dicts (e.g. "colors") so user additions
        # don't discard default entries.
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        return base

    available = ", ".join(p.stem for p in sorted(_PRESET_DIR.glob("*.json")) if p.stem != "named_colors")
    msg = f"Config not found: {name_or_path!r} (built-in presets: {available})"
    raise FileNotFoundError(msg)


def build_render_config(config_data: dict, cli_overrides: dict) -> RenderConfig:
    """Merge config dict with CLI overrides into a RenderConfig.

    ``config_data`` is the base layer (from JSON).
    ``cli_overrides`` contains only explicitly-set CLI values (non-None).
    CLI values win over config file values.
    """
    merged = {**config_data}
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v

    # "colors" key in JSON maps to color_overrides on RenderConfig
    colors = merged.pop("colors", None)
    if colors:
        merged["color_overrides"] = {sym: resolve_color(c) for sym, c in colors.items()}

    # MO/density keys are stored in config but not passed to RenderConfig
    # directly (they're used at build time, not render time). Strip them to
    # avoid TypeError from unexpected kwargs.
    merged.pop("mo_pos_color", None)
    merged.pop("mo_neg_color", None)
    merged.pop("mo_iso", None)
    merged.pop("mo_blur", None)
    merged.pop("mo_upsample", None)
    merged.pop("dens_iso", None)
    merged.pop("dens_color", None)

    # Resolve any named colors to hex for fields that downstream code parses as hex
    for key in ("background", "bond_color", "atom_stroke_color", "label_color", "cmap_unlabeled", "cell_color"):
        if key in merged:
            merged[key] = resolve_color(merged[key])

    # axis_colors comes from JSON as a list of 3 strings; convert to tuple and resolve colors
    if "axis_colors" in merged:
        raw = merged["axis_colors"]
        merged["axis_colors"] = tuple(resolve_color(c) for c in raw)

    return RenderConfig(**merged)
