import pytest

from meeting_transcriber.branding import footer_html, header_html, make_css, page_title, shades
from meeting_transcriber.config import load_config
from meeting_transcriber.errors import ConfigError


def _lum(h):
    return sum(int(h[i:i + 2], 16) for i in (1, 3, 5))


def test_shades_go_light_to_dark():
    s = shades("#0b3d91")
    order = ["c50", "c100", "c200", "c300", "c400", "c500", "c600", "c700", "c800", "c900", "c950"]
    lums = [_lum(s[k]) for k in order]
    assert lums == sorted(lums, reverse=True)
    assert s["c600"] == "#0b3d91"


def test_default_brand_and_title():
    cfg = load_config(env={})
    assert page_title(cfg) == "한미약품 AI 회의록"
    h = header_html(cfg)
    assert "한미약품" in h and "AI 회의록" in h and "외부 전송 없음" in h
    assert "사내 전용" in footer_html(cfg)
    assert cfg.ui.brand.primary_color in make_css(cfg)


def test_brand_is_configurable_and_escaped(tmp_path):
    cfg = load_config(env={"MT_UI__BRAND__APP_TITLE": "<script>x</script>회의록",
                           "MT_UI__BRAND__PRIMARY_COLOR": "#123456"})
    h = header_html(cfg)
    assert "<script>" not in h and "&lt;script&gt;" in h
    assert "#123456" in make_css(cfg)


def test_invalid_colour_rejected():
    with pytest.raises(ConfigError):
        load_config(env={"MT_UI__BRAND__PRIMARY_COLOR": "blue"})


def test_logo_embedded_when_configured(tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    cfg = load_config(env={"MT_UI__BRAND__LOGO_PATH": str(logo)})
    h = header_html(cfg)
    assert 'class="mt-logo"' in h and "data:image/png;base64," in h


def test_theme_builds():
    pytest.importorskip("gradio")
    from meeting_transcriber.branding import make_theme

    assert make_theme(load_config(env={})) is not None
