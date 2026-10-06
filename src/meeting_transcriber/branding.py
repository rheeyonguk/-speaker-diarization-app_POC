"""Corporate look & feel for the Gradio UI (colours, header, footer), driven by ``ui.brand`` config.

The default palette is a neutral pharma-style deep blue. Replace ``primary_color`` /
``accent_color`` with the official CI HEX values and point ``logo_path`` at the official logo file.
No web fonts are loaded (system Korean fonts only) so the UI works offline and makes no external requests.
"""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Optional

from .config import AppConfig

FONT_STACK = ["Pretendard", "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "Noto Sans KR", "system-ui", "sans-serif"]
MONO_STACK = ["D2Coding", "Consolas", "ui-monospace", "monospace"]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(c))) for c in rgb))


def mix(hex_color: str, other: str, t: float) -> str:
    """Blend ``hex_color`` towards ``other`` by ``t`` (0 = unchanged, 1 = other)."""
    a, b = _rgb(hex_color), _rgb(other)
    return _hex(tuple(x + (y - x) * t for x, y in zip(a, b)))


def shades(base: str) -> dict[str, str]:
    """11-step palette with ``base`` as the 600 shade (button colour)."""
    w, k = "#ffffff", "#000000"
    return {
        "c50": mix(base, w, 0.93), "c100": mix(base, w, 0.85), "c200": mix(base, w, 0.70),
        "c300": mix(base, w, 0.52), "c400": mix(base, w, 0.32), "c500": mix(base, w, 0.14),
        "c600": base, "c700": mix(base, k, 0.15), "c800": mix(base, k, 0.30),
        "c900": mix(base, k, 0.45), "c950": mix(base, k, 0.60),
    }


def make_theme(cfg: AppConfig):
    import gradio as gr

    b = cfg.ui.brand
    p = shades(b.primary_color)
    a = shades(b.accent_color)
    primary = gr.themes.Color(name="brand_primary", **p)
    accent = gr.themes.Color(name="brand_accent", **a)
    theme = gr.themes.Base(
        primary_hue=primary,
        secondary_hue=accent,
        neutral_hue="slate",
        radius_size="md",
        font=FONT_STACK,
        font_mono=MONO_STACK,
    )
    return theme.set(
        body_background_fill="#f3f6fa",
        block_background_fill="#ffffff",
        block_border_color="#e2e8f0",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.05)",
        block_radius="10px",
        block_label_text_color=p["c800"],
        block_title_text_color=p["c800"],
        background_fill_secondary=p["c50"],
        button_primary_background_fill=p["c600"],
        button_primary_background_fill_hover=p["c700"],
        button_primary_border_color=p["c600"],
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover=p["c50"],
        button_secondary_text_color=p["c700"],
        button_cancel_background_fill="#c62828",
        button_cancel_background_fill_hover="#a61f1f",
        button_cancel_text_color="#ffffff",
        button_large_radius="8px",
        input_radius="8px",
        input_border_color_focus=p["c500"],
        slider_color=p["c600"],
        checkbox_background_color_selected=p["c600"],
        checkbox_border_color_selected=p["c600"],
        color_accent=p["c600"],
        color_accent_soft=p["c50"],
        border_color_accent=p["c300"],
        border_color_primary=p["c200"],
        loader_color=a["c500"],
        link_text_color=p["c600"],
        table_even_background_fill="#ffffff",
        table_odd_background_fill=p["c50"],
    )


def make_css(cfg: AppConfig) -> str:
    b = cfg.ui.brand
    p = shades(b.primary_color)
    a = shades(b.accent_color)
    return f"""
.gradio-container {{ width: 100% !important; max-width: 1360px !important; margin: 0 auto !important; }}
.gradio-container .main {{ max-width: 1360px !important; }}
.mt-header {{
  background: linear-gradient(120deg, {p['c800']} 0%, {p['c600']} 58%, {a['c500']} 100%);
  color: #fff; border-radius: 14px; padding: 22px 28px; margin-bottom: 6px;
  box-shadow: 0 6px 18px rgba(11, 35, 80, 0.18);
}}
.mt-header, .mt-header * {{ color: #ffffff !important; }}
.mt-header .mt-top {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.mt-header .mt-logo {{ height: 38px; background: #fff; border-radius: 8px; padding: 4px 8px; }}
.mt-header .mt-company {{ font-size: 15px; font-weight: 700; letter-spacing: 0.08em; opacity: 0.95;
  border: 1px solid rgba(255,255,255,0.55); border-radius: 999px; padding: 3px 12px; }}
.mt-header .mt-title {{ font-size: 26px; font-weight: 800; letter-spacing: -0.01em; margin: 0; color: #fff; }}
.mt-header .mt-sub {{ margin-top: 8px; font-size: 14.5px; opacity: 0.92; }}
.mt-header .mt-badges {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
.mt-header .mt-badge {{ background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  border-radius: 999px; padding: 3px 11px; font-size: 12.5px; }}
.mt-header .mt-badge.mt-secure {{ background: rgba(255,255,255,0.94); color: {p['c800']} !important; font-weight: 700; }}
.mt-steps {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 2px; }}
.mt-steps .mt-step {{ flex: 1 1 160px; background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid {p['c600']};
  border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #334155; }}
.mt-steps .mt-step b {{ display: block; color: {p['c600']}; font-size: 11px; letter-spacing: 0.08em; margin-bottom: 2px; }}
.mt-section-title {{ font-size: 15px; font-weight: 700; color: {p['c800']}; margin: 6px 0 2px;
  padding-left: 8px; border-left: 3px solid {a['c500']}; }}
.mt-note {{ font-size: 13px; color: #475569; background: {p['c50']}; border: 1px solid {p['c100']};
  border-radius: 8px; padding: 10px 14px; }}
.mt-footer {{ text-align: center; color: #64748b; font-size: 12.5px; margin-top: 18px; padding-top: 12px;
  border-top: 1px solid #e2e8f0; }}
.mt-run button, button.mt-run {{ font-size: 16px !important; font-weight: 700 !important; min-height: 48px; }}
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {{ color: {p['c700']} !important;
  border-color: {p['c600']} !important; font-weight: 700; }}
"""


def _logo_data_uri(cfg: AppConfig) -> Optional[str]:
    path = cfg.ui.brand.logo_path
    if not path:
        return None
    p = cfg.resolve_path(path)
    if not p.is_file() or p.stat().st_size > 2_000_000:
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def page_title(cfg: AppConfig) -> str:
    b = cfg.ui.brand
    return f"{b.company} {b.app_title}".strip()


def header_html(cfg: AppConfig) -> str:
    b = cfg.ui.brand
    e = html.escape
    logo = _logo_data_uri(cfg)
    logo_html = f'<img class="mt-logo" src="{logo}" alt="{e(b.company)}"/>' if logo else ""
    company_html = f'<span class="mt-company">{e(b.company)}</span>' if b.company and not logo else ""
    badges = "".join(f'<span class="mt-badge">{e(x)}</span>' for x in b.badges)
    return f"""
<div class="mt-header">
  <div class="mt-top">{logo_html}{company_html}<h1 class="mt-title">{e(b.app_title)}</h1></div>
  <div class="mt-sub">{e(b.subtitle)}</div>
  <div class="mt-badges"><span class="mt-badge mt-secure">🔒 {e(b.security_badge)}</span>{badges}</div>
</div>"""


def steps_html() -> str:
    steps = [
        ("STEP 1", "참석자 음성 등록 (화자 등록 탭)"),
        ("STEP 2", "회의 음성·영상 업로드"),
        ("STEP 3", "참석 인원 설정 → 회의록 생성"),
        ("STEP 4", "검토 후 결과 파일 다운로드"),
    ]
    return '<div class="mt-steps">' + "".join(
        f'<div class="mt-step"><b>{n}</b>{html.escape(t)}</div>' for n, t in steps) + "</div>"


def footer_html(cfg: AppConfig) -> str:
    b = cfg.ui.brand
    parts = [p for p in (b.org_label, b.footer) if p]
    return f'<div class="mt-footer">{" · ".join(html.escape(p) for p in parts)}</div>'


def favicon_path(cfg: AppConfig) -> Optional[str]:
    p = cfg.ui.brand.favicon_path
    if not p:
        return None
    path = cfg.resolve_path(p)
    return str(path) if Path(path).is_file() else None
