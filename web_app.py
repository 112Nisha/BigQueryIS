"""Minimal web UI for generating and hosting citation tree HTML outputs."""

from __future__ import annotations

import importlib
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any
from werkzeug.utils import secure_filename

_flask = importlib.import_module("flask")
Flask = _flask.Flask
abort = _flask.abort
jsonify = _flask.jsonify
redirect = _flask.redirect
render_template = _flask.render_template
request = _flask.request
send_from_directory = _flask.send_from_directory
url_for = _flask.url_for

from citation_tree.config import OUTPUT_DIR
from citation_tree.main import build_trees

app = Flask(__name__)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
ALLOWED_OUTPUT_FILES = {
    "reference_tree.html",
    "citation_tree.html",
    "reference_tree.json",
    "citation_tree.json",
}


def _jobs_root() -> Path:
    root = Path(OUTPUT_DIR) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_output_dir(job_id: str) -> Path:
    path = _jobs_root() / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def _run_generation_job(job_id: str, source: str) -> None:
    _update_job(job_id, status="running")

    try:
        result = build_trees(
            source=source,
            output_dir=str(_job_output_dir(job_id)),
            open_browser=False,
        )

        has_reference = bool(result.get("has_reference"))
        has_citation = bool(result.get("has_citation"))

        if not has_reference and not has_citation:
            _update_job(
                job_id,
                status="failed",
                error="Generation finished but no HTML output was produced.",
            )
            return

        _update_job(
            job_id,
            status="completed",
            has_reference=has_reference,
            has_citation=has_citation,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/jobs")
def create_job():
    uploaded_pdf = request.files.get("pdf_file")
    if uploaded_pdf is None or not (uploaded_pdf.filename or "").strip():
        return render_template("index.html", error="Please upload a PDF file."), 400

    original_name = secure_filename(uploaded_pdf.filename)
    if not original_name or not original_name.lower().endswith(".pdf"):
        return render_template("index.html", error="Only .pdf uploads are supported."), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_output_dir(job_id)
    source_path = job_dir / "input.pdf"

    try:
        uploaded_pdf.save(source_path)
    except Exception:
        return render_template("index.html", error="Could not save the uploaded PDF."), 400

    if not source_path.exists() or source_path.stat().st_size == 0:
        return render_template("index.html", error="Uploaded PDF is empty."), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "source": original_name,
            "status": "queued",
            "has_reference": False,
            "has_citation": False,
            "error": None,
            "traceback": None,
        }

    worker = threading.Thread(
        target=_run_generation_job,
        args=(job_id, str(source_path)),
        daemon=True,
    )
    worker.start()

    return redirect(url_for("job_status", job_id=job_id))


@app.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        abort(404)
    return render_template("status.html", job=job)


@app.get("/api/jobs/<job_id>")
def job_status_api(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    payload = {
        "id": job["id"],
        "source": job["source"],
        "status": job["status"],
        "error": job.get("error"),
        "has_reference": bool(job.get("has_reference")),
        "has_citation": bool(job.get("has_citation")),
    }

    if payload["status"] == "completed":
        payload["result_url"] = url_for("job_result", job_id=job_id)

    return jsonify(payload)


@app.get("/jobs/<job_id>/result")
def job_result(job_id: str):
    job = _get_job(job_id)
    if not job:
        abort(404)

    if job["status"] != "completed":
        return redirect(url_for("job_status", job_id=job_id))

    view = (request.args.get("view") or "").strip().lower()
    if view not in {"reference_tree.html", "citation_tree.html"}:
        view = "reference_tree.html" if job.get("has_reference") else "citation_tree.html"

    if view == "reference_tree.html" and not job.get("has_reference"):
        view = "citation_tree.html"
    if view == "citation_tree.html" and not job.get("has_citation"):
        view = "reference_tree.html"

    return render_template(
        "result.html",
        job=job,
        active_view=view,
        active_view_url=url_for("job_output_file", job_id=job_id, filename=view),
    )


@app.get("/jobs/<job_id>/files/<path:filename>")
def job_output_file(job_id: str, filename: str):
    if filename not in ALLOWED_OUTPUT_FILES:
        abort(404)

    job = _get_job(job_id)
    if not job:
        abort(404)

    output_dir = _job_output_dir(job_id)
    target = output_dir / filename
    if not target.exists() or not target.is_file():
        abort(404)

    return send_from_directory(output_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
