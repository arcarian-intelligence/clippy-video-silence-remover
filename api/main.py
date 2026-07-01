"""FastAPI backend for Clippy — Video Silence Remover."""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List
import os
import signal
import shutil
import subprocess
import threading
import time
import zipfile

from processor import jobs, create_job, run_job, create_batch_job, run_batch_job, cleanup_old_jobs, JOBS_DIR

app = FastAPI(title="Clippy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 5000


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    threshold: int = Form(-50),
    start_padding: int = Form(30),
    end_padding: int = Form(50),
    min_silence: int = Form(300),
):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Clean up old jobs on each upload
    cleanup_old_jobs()

    job_id = create_job(file.filename)
    job = jobs[job_id]

    # Save uploaded file to disk
    with open(job["input_path"], "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = Path(job["input_path"]).stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_MB:
        Path(job["input_path"]).unlink()
        jobs.pop(job_id, None)
        raise HTTPException(413, f"File too large ({file_size_mb:.0f}MB). Max is {MAX_UPLOAD_MB}MB.")

    # Start processing in background
    run_job(job_id, threshold, start_padding, end_padding, min_silence)

    return {"job_id": job_id}


@app.post("/api/upload-batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    threshold: int = Form(-50),
    start_padding: int = Form(30),
    end_padding: int = Form(50),
    min_silence: int = Form(300),
    order: str = Form(""),
):
    if not files:
        raise HTTPException(400, "No files provided")

    # Clean up old jobs
    cleanup_old_jobs()

    # Parse the order string (comma-separated indices)
    if order:
        indices = [int(i) for i in order.split(",")]
        files = [files[i] for i in indices]

    filenames = [f.filename or f"video_{i}.mp4" for i, f in enumerate(files)]
    job_id = create_batch_job(filenames)
    job = jobs[job_id]

    # Save all uploaded files
    total_size = 0
    for i, file in enumerate(files):
        with open(job["input_paths"][i], "wb") as f:
            shutil.copyfileobj(file.file, f)
        total_size += Path(job["input_paths"][i]).stat().st_size

    total_size_mb = total_size / (1024 * 1024)
    if total_size_mb > MAX_UPLOAD_MB:
        for p in job["input_paths"]:
            Path(p).unlink(missing_ok=True)
        jobs.pop(job_id, None)
        raise HTTPException(413, f"Total size too large ({total_size_mb:.0f}MB). Max is {MAX_UPLOAD_MB}MB.")

    # Start batch processing
    run_batch_job(job_id, threshold, start_padding, end_padding, min_silence)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    response = {
        "status": job["status"],
        "step": job["step"],
        "segments": job["segments"],
        "input_size_mb": job.get("input_size_mb"),
        "output_size_mb": job.get("output_size_mb"),
        "segment_count": len(job["segment_files"]) if job.get("segment_files") else None,
    }

    # Include batch-specific fields
    if job.get("batch"):
        response["batch"] = True
        response["total_files"] = job["total_files"]
        response["current_file"] = job.get("current_file", 0)

        # Include individual file info when done
        if job["status"] == "done" and job.get("trimmed_files"):
            response["trimmed_files"] = [
                {
                    "index": i,
                    "filename": tf["original_filename"],
                    "size_mb": tf["size_mb"],
                }
                for i, tf in enumerate(job["trimmed_files"])
            ]

    return response


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Video not ready yet")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    if job.get("batch"):
        download_name = "clippy_merged.mp4"
    else:
        original = job["original_filename"]
        stem = Path(original).stem
        download_name = f"{stem}_trimmed.mp4"

    return FileResponse(
        path=str(output_path),
        filename=download_name,
        media_type="video/mp4",
    )


@app.get("/api/download/{job_id}/{file_index}")
async def download_individual(job_id: str, file_index: int):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Video not ready yet")

    if not job.get("trimmed_files"):
        raise HTTPException(400, "No individual files available")

    if file_index < 0 or file_index >= len(job["trimmed_files"]):
        raise HTTPException(404, "File index out of range")

    trimmed = job["trimmed_files"][file_index]
    file_path = Path(trimmed["path"])
    if not file_path.exists():
        raise HTTPException(404, "Trimmed file not found")

    stem = Path(trimmed["original_filename"]).stem
    download_name = f"{stem}_trimmed.mp4"

    return FileResponse(
        path=str(file_path),
        filename=download_name,
        media_type="video/mp4",
    )


@app.get("/api/download-zip/{job_id}")
async def download_zip(job_id: str, indices: str = Query("")):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Videos not ready yet")

    if not job.get("trimmed_files"):
        raise HTTPException(400, "No individual files available")

    # Parse which files to include (empty = all)
    trimmed = job["trimmed_files"]
    if indices:
        selected = [int(i) for i in indices.split(",")]
        selected = [i for i in selected if 0 <= i < len(trimmed)]
    else:
        selected = list(range(len(trimmed)))

    if not selected:
        raise HTTPException(400, "No valid files selected")

    # Build zip in the job directory
    from processor import JOBS_DIR
    zip_path = JOBS_DIR / job_id / "clippy_trimmed_clips.zip"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_STORED) as zf:
        for i in selected:
            tf = trimmed[i]
            file_path = Path(tf["path"])
            if file_path.exists():
                stem = Path(tf["original_filename"]).stem
                zf.write(str(file_path), f"{stem}_trimmed.mp4")

    return FileResponse(
        path=str(zip_path),
        filename="clippy_trimmed_clips.zip",
        media_type="application/zip",
    )


@app.get("/api/download-segments/{job_id}")
async def download_segments(job_id: str):
    """Download all speech segments as numbered clips (001.mp4, 002.mp4, ...) in a zip."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Video not ready yet")

    seg_files = job.get("segment_files")
    if not seg_files:
        raise HTTPException(400, "No segment files available")

    zip_path = JOBS_DIR / job_id / "clippy_timeline_clips.zip"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_STORED) as zf:
        for sf in seg_files:
            file_path = Path(sf["path"])
            if file_path.exists():
                zf.write(str(file_path), file_path.name)

    return FileResponse(
        path=str(zip_path),
        filename="clippy_timeline_clips.zip",
        media_type="application/zip",
    )


@app.post("/api/shutdown")
async def shutdown():
    """Kill the frontend, wipe temp files, and shut down the API itself."""
    threading.Thread(target=_shutdown_sequence, daemon=True).start()
    return {"status": "shutting down"}


CLIPPY_ROOT = str(Path(__file__).resolve().parent.parent)


def _pid_cwd(pid: int) -> str | None:
    """Return the cwd of a PID via lsof, or None if it can't be read."""
    try:
        out = subprocess.run(
            ["lsof", "-p", str(pid), "-a", "-d", "cwd", "-F", "n"],
            capture_output=True, text=True, check=False, timeout=2,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def _find_clippy_pids() -> list[int]:
    """PIDs of Clippy-owned processes (uvicorn / next / node running next),
    scoped to those whose cwd lives inside the Clippy project directory."""
    result = subprocess.run(
        ["ps", "-eo", "pid,args"],
        capture_output=True, text=True, check=False,
    )
    tokens = ("uvicorn", "next-server", "next dev", "next start", "node_modules/.bin/next")
    pids: list[int] = []
    for line in result.stdout.strip().splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, cmd = parts
        if not any(tok in cmd for tok in tokens):
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        cwd = _pid_cwd(pid)
        if cwd and cwd.startswith(CLIPPY_ROOT):
            pids.append(pid)
    return pids


def _shutdown_sequence() -> None:
    # Let the HTTP response flush before we start killing things.
    time.sleep(0.5)

    self_pid = os.getpid()
    parent_pid = os.getppid()

    clippy_pids = _find_clippy_pids()
    others = [p for p in clippy_pids if p not in (self_pid, parent_pid)]

    # First pass: SIGTERM to give processes a chance to clean up.
    for pid in others:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(0.5)

    # Second pass: SIGKILL any that stuck around.
    for pid in others:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Wipe tmp contents (keep the directory itself).
    for item in JOBS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)

    # Finally, kill the uvicorn supervisor (parent) and this worker (self).
    try:
        os.kill(parent_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    os.kill(self_pid, signal.SIGTERM)
