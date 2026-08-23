"""Exports must produce valid, non-trivial PPTX / DOCX / PDF from a full run."""
import io
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401
import contextlib

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from agents.narrator import run_narration
from agents.exporter import build_pptx, build_docx, build_pdf, render_charts
from tests.make_demo_data import make_utilities_csv

csv = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
make_utilities_csv(csv)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="find revenue leakage, forecast next month, "
                        "explain revenue movement, flag billing anomalies, segment customers, run what-if "
                        "scenarios and recommend actions",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_exp"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)
    summary = run_narration(st, use_llm=False)

charts = render_charts(st)
assert {"forecast", "bakeoff", "anomaly", "leakage"} <= set(charts), \
    f"charts missing: {set(charts)}"

base = Path(tempfile.gettempdir())
pptx_p = build_pptx(st, summary, str(base / "t.pptx"))
docx_p = build_docx(st, summary, str(base / "t.docx"))
pdf_p = build_pdf(st, summary, str(base / "t.pdf"))

# pptx: valid zip, slide count sane
zf = zipfile.ZipFile(pptx_p)
slides = [n for n in zf.namelist() if n.startswith("ppt/slides/slide")]
assert len(slides) >= 6, f"only {len(slides)} slides"
assert zf.testzip() is None

# docx: valid zip, has document + at least one embedded image
zf = zipfile.ZipFile(docx_p)
assert "word/document.xml" in zf.namelist()
assert any(n.startswith("word/media/") for n in zf.namelist()), "no charts embedded"
doc_xml = zf.read("word/document.xml").decode("utf-8", "replace")
assert "accountability" in doc_xml.lower()

# pdf: magic bytes + multi-page (rough size check)
pdf_bytes = Path(pdf_p).read_bytes()
assert pdf_bytes[:5] == b"%PDF-"
assert pdf_bytes.count(b"/Type /Page") >= 4 or pdf_bytes.count(b"/Type/Page") >= 4

print(f"pptx {len(slides)} slides | docx with charts | pdf "
      f"{max(pdf_bytes.count(b'/Type /Page'), pdf_bytes.count(b'/Type/Page'))} pages")
print("EXPORT CHECKS PASSED ✅")
