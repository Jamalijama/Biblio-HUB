import uuid
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from modules.export_orchestrator import generate_export_zip, generate_master_export
from modules.figure_export_bundle import generate_all_figure_bundle
from modules.copyright_materials import generate_copyright_package

                                     
_export_executor = ThreadPoolExecutor(max_workers=1)
_export_jobs: dict[str, dict] = {}

def submit_export_background_job(job_type, *args, **kwargs):
    """
    Submits an export task to the background executor.
    job_type: 'full_export', 'figure_bundle', or 'copyright_package'
    """
    job_id = str(uuid.uuid4())
    
    if job_type == 'full_export':
        future = _export_executor.submit(generate_export_zip, *args, **kwargs)
    elif job_type == 'master_export':
        future = _export_executor.submit(generate_master_export, *args, **kwargs)
    elif job_type == 'figure_bundle':
        future = _export_executor.submit(generate_all_figure_bundle, *args, **kwargs)
    elif job_type == 'copyright_package':
        future = _export_executor.submit(generate_copyright_package, *args, **kwargs)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
        
    _export_jobs[job_id] = {
        "future": future,
        "status": "running",
        "type": job_type,
        "result": None,
        "error": None
    }
    return job_id

def get_export_job_status(job_id):
    """
    Checks the status of a background export job.
    """
    if job_id not in _export_jobs:
        return {"status": "not_found"}
        
    job = _export_jobs[job_id]
    if job["status"] == "running":
        if job["future"].done():
            try:
                job["result"] = job["future"].result()
                job["status"] = "completed"
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                
    return {
        "status": job["status"],
        "type": job["type"],
        "result": job["result"],
        "error": job["error"]
    }

def discard_export_job(job_id):
    """
    Removes a job from the registry.
    """
    if job_id in _export_jobs:
        del _export_jobs[job_id]
