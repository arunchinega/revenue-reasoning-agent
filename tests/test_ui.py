"""UI smoke test via streamlit.testing.AppTest.

The file-uploader can't be driven headlessly, so we pre-build a RunState
(perception already done) and inject it, then drive: plan-approval click →
capability execution → results tabs. Covers every phase except the upload
widget itself.
"""
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from streamlit.testing.v1 import AppTest

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from agents.narrator import run_narration
from tests.make_demo_data import make_utilities_csv

NL = ("find revenue leakage, forecast next month, explain why revenue moved, "
      "segment my customers, run what-if scenarios and recommend actions")


def _perceived_state():
    csv = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv)
    return run_perception(csv, nl_request=NL,
                          base_dir=str(Path(tempfile.gettempdir()) / "rra_ui"),
                          use_llm=False)


def test_confirm_to_done() -> None:
    state = _perceived_state()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                           default_timeout=1800)
    at.session_state["phase"] = "confirm"
    at.session_state["state"] = state
    at.session_state["sync_run"] = True
    at.run()
    assert not at.exception, at.exception
    assert any("Confirm the plan" in str(h.value) for h in at.subheader), \
        "confirm card missing"

    # click ✅ Approve plan & run — fall-through phases mean this single run
    # goes confirm→execute→done with no st.rerun() hops (1.5x-AppTest-safe)
    approve = [b for b in at.button if "Approve" in b.label]
    assert approve, "approve button missing"
    approve[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["phase"] == "done"
    assert "narration" in at.session_state["state"].results
    assert any("Analysis complete" in str(x.value) for x in at.success), \
        "done banner missing on Run screen"
    print("phases confirm→execute→done OK; "
          f"ledger entries: {len(at.session_state['state'].ledger.entries)}")


def test_screens_render() -> None:
    """Each screen renders against a fully-populated state — a fresh AppTest
    per screen (1.5x AppTest can't switch screens with vanishing widgets)."""
    state = _perceived_state()
    run_capabilities(state, use_llm=False)
    summary = run_narration(state, use_llm=False)
    app = str(Path(__file__).resolve().parents[1] / "app.py")

    for screen, must_contain in [
        ("📖 Findings", "Money at risk"),
        ("📊 Data & Features", "Feature engineering"),
        ("🔬 Evidence", "Forecast"),
        ("🧠 Reasoning", "decisions"),
        ("📄 Export", "Export"),
    ]:
        at = AppTest.from_file(app, default_timeout=600)
        at.session_state["phase"] = "done"
        at.session_state["state"] = state
        at.session_state["summary"] = summary
        at.session_state["nav"] = screen
        at.run()
        assert not at.exception, f"{screen}: {at.exception}"
        page_text = " ".join(str(m.value) for m in at.markdown) + " " + \
            " ".join(str(m.label) for m in at.metric) + " " + \
            " ".join(str(t.value) for t in at.title) + " " + \
            " ".join(str(x.label) for x in at.expander)
        assert must_contain.lower() in page_text.lower(), \
            f"{screen} missing '{must_contain}'"
        print(f"{screen} renders OK")


def test_perceive_without_file_self_heals() -> None:
    """A rerun into phase='perceive' with no uploaded file must reset to idle
    with a message — not crash on up.name (the AttributeError Arun hit)."""
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                           default_timeout=120)
    at.session_state["phase"] = "perceive"
    at.session_state["sync_run"] = True
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["phase"] == "idle"
    assert any("upload" in str(i.value).lower() for i in at.info), \
        "self-heal message missing"
    print("perceive-without-file self-heal OK")


if __name__ == "__main__":
    test_perceive_without_file_self_heals()
    test_confirm_to_done()
    test_screens_render()
    print("UI CHECKS PASSED ✅")
