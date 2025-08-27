"""
Crawler Dashboard - Web interface for managing crawler jobs
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import json
from typing import Dict, List
import os

# Create dashboard app
dashboard_app = FastAPI(
    title="Crawler Management Dashboard",
    description="Web interface for managing crawler scheduling",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="templates")

# API base URL (where your main API runs)
API_BASE_URL = "http://localhost:8000"

@dashboard_app.get("/crawler", response_class=HTMLResponse)
async def crawler_dashboard(request: Request):
    """Main crawler dashboard page"""
    return templates.TemplateResponse("crawler_dashboard.html", {"request": request})

@dashboard_app.get("/api/crawler/jobs")
async def get_jobs():
    """Proxy to get crawler jobs"""
    try:
        response = requests.get(f"{API_BASE_URL}/crawler/jobs", timeout=10)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {e}")

@dashboard_app.get("/api/crawler/status")
async def get_status():
    """Proxy to get crawler status"""
    try:
        response = requests.get(f"{API_BASE_URL}/crawler/status", timeout=10)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {e}")

@dashboard_app.post("/api/crawler/jobs")
async def create_job(job_data: dict):
    """Proxy to create crawler job"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/crawler/jobs", 
            json=job_data, 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {e}")

@dashboard_app.post("/api/crawler/jobs/{job_id}/run")
async def run_job(job_id: str):
    """Proxy to run crawler job"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/crawler/jobs/{job_id}/run", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run job: {e}")

@dashboard_app.put("/api/crawler/jobs/{job_id}/enable")
async def enable_job(job_id: str):
    """Proxy to enable crawler job"""
    try:
        response = requests.put(
            f"{API_BASE_URL}/crawler/jobs/{job_id}/enable", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable job: {e}")

@dashboard_app.put("/api/crawler/jobs/{job_id}/disable")
async def disable_job(job_id: str):
    """Proxy to disable crawler job"""
    try:
        response = requests.put(
            f"{API_BASE_URL}/crawler/jobs/{job_id}/disable", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable job: {e}")

@dashboard_app.delete("/api/crawler/jobs/{job_id}")
async def delete_job(job_id: str):
    """Proxy to delete crawler job"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/crawler/jobs/{job_id}", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {e}")

@dashboard_app.get("/api/crawler/jobs/{job_id}/results")
async def get_job_results(job_id: str, limit: int = 20):
    """Proxy to get job results"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/crawler/jobs/{job_id}/results?limit={limit}", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {e}")

@dashboard_app.post("/api/crawler/jobs/create-defaults")
async def create_defaults():
    """Proxy to create default jobs"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/crawler/jobs/create-defaults", 
            timeout=10
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create defaults: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Crawler Management Dashboard...")
    print("Access at: http://localhost:8081/crawler")
    
    uvicorn.run(
        "crawler_dashboard:dashboard_app",
        host="0.0.0.0",
        port=8081,
        reload=True
    )