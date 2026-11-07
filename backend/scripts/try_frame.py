"""Send a photo (or a folder of photos) through the pipeline from a laptop.
Great for tuning prompts and for a backup demo if the phone misbehaves.

    python scripts/try_frame.py photos/corridor.jpg --goal "Take me to Room 443B"
    python scripts/try_frame.py photos/ --mode describe
"""
import argparse
import json
import pathlib
import sys

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--goal", help='voice command to send first, e.g. "Take me to Room 443B"')
    ap.add_argument("--mode", default="auto", choices=["auto", "describe", "read"])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    p = pathlib.Path(a.path)
    files = sorted(x for x in (p.iterdir() if p.is_dir() else [p]) if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        sys.exit("No images found")

    with httpx.Client(base_url=a.api, timeout=30) as c:
        sid = c.post("/v1/sessions").json()["session_id"]
        if a.goal:
            r = c.post(f"/v1/sessions/{sid}/command", json={"text": a.goal}).json()
            print("GOAL >", " ".join(u["text"] for u in r["utterances"]))
        for f in files:
            r = c.post(f"/v1/sessions/{sid}/frame", data={"mode": a.mode},
                       files={"image": (f.name, f.read_bytes(), "image/jpeg")}).json()
            print(f"\n[{f.name}] phase={r['phase']} arrival={r['arrival']['state']} timings={r['timings_ms']}")
            for u in r["utterances"]:
                print(f"  SAY({u['priority']}) {u['text']}")
            if a.verbose:
                print(json.dumps({k: r[k] for k in ("scene", "hazards", "ocr")}, indent=2))


if __name__ == "__main__":
    main()
