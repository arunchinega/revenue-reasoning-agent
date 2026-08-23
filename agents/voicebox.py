"""
agents/voicebox.py — server-side narration. Synthesises the whole story into
ONE wav (female narrator / male critic per line on Windows SAPI), played via
st.audio — the browser speech engine is never touched.
Fails soft: returns None on any error so the UI can fall back.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

FEMALE = ("zira", "heera", "neerja", "female", "hazel", "susan")
MALE = ("david", "ravi", "mark", "male", "george")


def _pick_voice(engine, hints: tuple[str, ...]):
    try:
        for v in engine.getProperty("voices"):
            name = (v.name or "").lower()
            if any(h in name for h in hints):
                return v.id
    except Exception:  # noqa: BLE001
        pass
    return None


def synth_narration(items: list[tuple[str, str | None]],
                    duet: bool = True) -> bytes | None:
    """items = [(text, speaker)] with speaker in {narrator, critic, None}."""
    try:
        import pyttsx3
        tmp = Path(tempfile.mkdtemp(prefix="rra_voice_"))
        segs: list[Path] = []
        for i, (text, speaker) in enumerate(items):
            clean = (text.replace("**", "").replace("`", "").replace("#", "")
                     .replace("→", " to ").replace("Δ", "delta "))
            if not clean.strip():
                continue
            engine = pyttsx3.init()
            engine.setProperty("rate", 178)
            hints = MALE if (duet and speaker == "critic") else FEMALE
            vid = _pick_voice(engine, hints)
            if vid:
                engine.setProperty("voice", vid)
            seg = tmp / f"seg{i:02d}.wav"
            engine.save_to_file(clean, str(seg))
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            if seg.exists() and seg.stat().st_size > 200:
                segs.append(seg)
        if not segs:
            return None
        out = tmp / "narration.wav"
        with wave.open(str(segs[0]), "rb") as w0:
            params = w0.getparams()
        with wave.open(str(out), "wb") as wf:
            wf.setparams(params)
            gap = b"\x00" * int(params.framerate * params.sampwidth *
                                params.nchannels * 0.35)
            for s in segs:
                with wave.open(str(s), "rb") as wr:
                    if wr.getparams()[:3] != params[:3]:
                        continue
                    wf.writeframes(wr.readframes(wr.getnframes()))
                wf.writeframes(gap)
        return out.read_bytes()
    except Exception:  # noqa: BLE001
        return None
