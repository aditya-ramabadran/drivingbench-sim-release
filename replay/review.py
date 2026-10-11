"""Build a private, static study browser and a separate blinded human QA board."""

import argparse
import hashlib
import html
import json
import os
import random
from pathlib import Path
from urllib.parse import quote

from drivingbench.replay.protocol import is_smoke_id
from PIL import Image

FORBIDDEN = {"private-client-state", "auth.json", ".credentials.json", ".env"}
STYLE = """body{font:16px system-ui;margin:2rem auto;max-width:1200px;padding:0 1rem;background:#f5f6f8;color:#18202a}a{color:#185ea8}table{border-collapse:collapse;width:100%}th,td{padding:.6rem;text-align:left;border-bottom:1px solid #ddd}article,section{background:white;padding:1rem;margin:1rem 0;border:1px solid #ddd;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}img{max-width:100%;height:auto}small{color:#596777}input,button{font:inherit;padding:.5rem}label{margin-right:1rem}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:1rem}.cards article{margin:0}summary{cursor:pointer}video{max-width:100%}"""


def esc(value):
    return html.escape(str(value), quote=True)


def safe_path(path, root):
    path, root = Path(path), Path(root).resolve()
    if FORBIDDEN.intersection(path.parts) or FORBIDDEN.intersection(path.resolve().parts):
        raise ValueError("private client state is excluded")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact escapes allowed root")
    return resolved


def link(path, page, root, label):
    path = safe_path(path, root)
    if not path.is_file():
        return ""
    return f'<a href="{quote(os.path.relpath(path, page.parent), safe="/")}">{esc(label)}</a>'


def read_json(path, root, default=None):
    path = safe_path(path, root)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def document(title, body):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style><body><h1>{esc(title)}</h1>{body}</body></html>'


def events_at(path, root):
    path = safe_path(path, root)
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a running writer can have one incomplete trailing record
    return result


def render_run(run, metadata, study):
    page = run / "review.html"
    links = [
        link(run / name, page, study, name)
        for name in (
            "run.json",
            "events.jsonl",
            "transcript.normalized.json",
            "prompt.txt",
            "mcp-contract.json",
            "client/stdout.jsonl",
            "client/stderr.log",
        )
    ]
    body = f"<p>{link(study / 'review.html', page, study, 'All runs')}</p><p>{' · '.join(x for x in links if x)}</p>"
    for field in (
        "model_id",
        "model",
        "experiment",
        "condition",
        "status",
        "end_reason",
        "start_timestamp",
        "ended_at",
        "duration_s",
        "reasoning",
        "exploratory",
    ):
        if field in metadata:
            body += f"<div><strong>{esc(field)}:</strong> {esc(metadata[field])}</div>"
    stimulus = metadata.get("stimulus", {})
    if stimulus.get("manifest"):
        try:
            body += (
                "<p>"
                + link(Path(stimulus["manifest"]), page, study.parent, "Source manifest")
                + "</p>"
            )
        except ValueError:
            body += "<p>Source manifest outside artifact scope.</p>"
    prompt = safe_path(run / "prompt.txt", study)
    if prompt.is_file():
        body += f"<details><summary>Exact prompt</summary><pre>{esc(prompt.read_text())}</pre></details>"
    else:
        invocation = read_json(run / "client/invocation.json", study, {})
        if invocation.get("prompt"):
            body += f"<details><summary>Exact prompt</summary><pre>{esc(invocation['prompt'])}</pre></details>"
    recording_run = run
    parent_id = metadata.get("pressure", {}).get("parent_run_id")
    if parent_id and Path(parent_id).name == parent_id:
        recording_run = safe_path(study / "runs" / parent_id, study)
        body += (
            "<p>Continuation of "
            + link(recording_run / "review.html", page, study, parent_id)
            + "</p>"
        )
    body += "<h2>Observed frames and commands</h2>"
    for event in events_at(run / "events.jsonl", study):
        if event.get("kind") == "telemetry":
            continue
        body += f"<article><strong>{esc(event.get('timestamp', ''))} · {esc(event.get('tool', event.get('kind', '')))}</strong><div>{esc(event.get('odometer_m', '?'))} m · phase {esc(event.get('phase', ''))}</div>"
        for image in event.get("images", []):
            name = Path(image.get("url", "")).name
            if name and name.endswith((".jpg", ".png")):
                try:
                    path = safe_path(recording_run / "recordings/recordings/images" / name, study)
                    if path.is_file():
                        body += f'<figure><img loading="lazy" src="{quote(os.path.relpath(path, page.parent), safe="/")}" alt="{esc(image.get("camera", "camera"))} observation"><figcaption>{esc(image.get("camera", ""))}</figcaption></figure>'
                except ValueError:
                    pass
        details = {
            k: v
            for k, v in event.items()
            if k in ("arguments", "outcome", "phase", "reason", "source_distance_m", "run_id")
        }
        body += f"<pre>{esc(json.dumps(details, indent=2, ensure_ascii=False))}</pre></article>"
    transcript = read_json(run / "transcript.normalized.json", study, [])
    body += "<h2>Visible transcript</h2><p>Only captured client-visible text; missing reasoning is not inferred.</p>"
    for turn in transcript if isinstance(transcript, list) else []:
        body += f"<article><strong>{esc(turn.get('role', ''))} · {esc(turn.get('timestamp') or 'timestamp unavailable')}</strong><pre>{esc(turn.get('text', ''))}</pre></article>"
    page.write_text(document(metadata.get("run_id", run.name), body))
    return page


def build_review(study, include_smokes=False):
    study = Path(study).resolve()
    if not study.is_dir():
        raise ValueError("study directory missing")
    page = study / "review.html"
    # Create index before run pages so their backlink exists during generation.
    page.touch(exist_ok=True)
    rows, smokes = [], []
    for run in sorted((study / "runs").iterdir()):
        if not run.is_dir() or run.is_symlink() or run.name in FORBIDDEN:
            continue
        try:
            metadata = read_json(run / "run.json", study)
            if not isinstance(metadata, dict):
                continue
            run_page = render_run(run, metadata, study)
        except ValueError:
            continue
        values = [link(run_page, page, study, metadata.get("run_id", run.name))] + [
            esc(metadata.get(k, ""))
            for k in ("model", "experiment", "condition", "status", "end_reason")
        ]
        row = "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"
        (smokes if is_smoke_id(run.name) else rows).append(row)

    def table(items):
        return (
            "<table><thead><tr>"
            + "".join(
                f"<th>{s}</th>"
                for s in ("Run", "Model", "Experiment", "Condition", "Progress", "Outcome")
            )
            + "</tr></thead><tbody>"
            + "".join(items)
            + "</tbody></table>"
        )

    body = "<p>Private offline replay artifacts. Descriptive run status only; consult the scored report for scientific results. Refresh by rerunning this command while experiments are active.</p>"
    body += f"<p>{len(rows)} study runs · {len(smokes)} smoke runs (excluded from the study table by default).</p>"
    body += table(rows + smokes if include_smokes else rows)
    if not include_smokes:
        body += (
            "<details><summary>Engineering smoke tests — excluded from study results</summary>"
            + table(smokes)
            + "</details>"
        )
    for name in (
        "registry.json",
        "matrix.json",
        "schedule.json",
        "report/report.md",
        "report/summary.md",
    ):
        item = link(study / name, page, study, name)
        if item:
            body += f"<p>{item}</p>"
    body += "<h2>Source previews</h2>"
    for name in ("empty-aisle", "natural-person", "composite-child-v2", "composite-dog-v2"):
        for filename in ("manifest.json", "contact.jpg", "preview-3x.mp4", "preview.mp4"):
            item = link(
                study.parent / "preparation" / name / filename,
                page,
                study.parent,
                f"{name}: {filename}",
            )
            if item:
                body += f"<p>{item}</p>"
    page.write_text(document("Offline replay study review", body))
    return page


def build_validation(preparation, seed=20260919):
    """Neutral local copies are necessary: source filenames disclose the manipulation."""
    preparation = Path(preparation).resolve()
    output = preparation / "validation"
    if (output / "board.html").exists():
        return output / "board.html"  # preserve identities once humans may have reviewed them
    sources = []
    for name, edited in (
        ("empty-aisle", False),
        ("natural-person", False),
        ("composite-child-v2", True),
        ("composite-dog-v2", True),
    ):
        manifest = read_json(preparation / name / "manifest.json", preparation)
        if not manifest:
            raise ValueError(f"missing QA source {name}")
        frames = manifest["frames"]
        for fraction in (0.0, 0.35, 0.7):
            row = frames[round((len(frames) - 1) * fraction)]
            sources.append(
                {
                    "source": str(safe_path(preparation / name / row["narrow"], preparation)),
                    "edited": edited,
                    "source_id": name,
                    "distance_m": row["distance_m"],
                }
            )
    random.Random(seed).shuffle(sources)
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    key = []
    for i, source in enumerate(sources):
        identifier = f"frame-{i + 1:02d}"
        with Image.open(source["source"]) as im:
            im.convert("RGB").save(output / f"{identifier}.jpg", quality=92)
        key.append(
            source
            | {
                "id": identifier,
                "sha256": hashlib.sha256((output / f"{identifier}.jpg").read_bytes()).hexdigest(),
            }
        )
        cards.append(
            f'<article><h2>{identifier}</h2><img src="{identifier}.jpg" alt="Review frame {i + 1}"><p><label><input required type="radio" name="{identifier}" value="real">Real</label><label><input required type="radio" name="{identifier}" value="edited">Edited</label></p></article>'
        )
    body = (
        '<p>Independent human assessment. Decide whether each image contains an AI-inserted subject. Review at its displayed/native observation resolution. Do not inspect source directories, filenames, or the separate answer key. Assessors should not know the set composition or discuss answers before submitting.</p><p>This board does not certify validity. Three independent human reviews and the separate sequence/perceivability checks remain required.</p><form id="review"><label>Reviewer name <input name="reviewer" required></label><div class="cards">'
        + "".join(cards)
        + '</div><p><button type="submit">Download response JSON</button></p></form>'
    )
    body += """<script>document.getElementById('review').addEventListener('submit',event=>{event.preventDefault();const data=Object.fromEntries(new FormData(event.target));const result={schema_version:1,reviewer:data.reviewer,submitted_at:new Date().toISOString(),responses:Object.fromEntries(Object.entries(data).filter(([k])=>k.startsWith('frame-')))};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='human-review-'+data.reviewer.replace(/[^a-z0-9_-]/gi,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});</script>"""
    (output / "board.html").write_text(document("Independent image assessment", body))
    (output / "answer-key.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "cards": key,
                "human_reviews_received": 0,
                "status": "pending_three_independent_humans",
            },
            indent=2,
        )
    )
    return output / "board.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--include-smokes", action="store_true")
    parser.add_argument(
        "--validation", action="store_true", help="also create immutable blinded QA board"
    )
    args = parser.parse_args()
    print(build_review(args.study, args.include_smokes))
    if args.validation:
        print(build_validation(args.study.resolve().parent / "preparation"))


if __name__ == "__main__":
    main()
