#!/usr/bin/env python3
"""
Background job management system for threshold tuning tasks
"""

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    task_type: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    current_message: str = ""
    result_file: Optional[str] = None
    error_message: Optional[str] = None
    request_params: Optional[Dict] = None
    
    def to_dict(self):
        """Convert job to dictionary for JSON serialization"""
        job_dict = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if job_dict[field]:
                job_dict[field] = job_dict[field].isoformat()
        return job_dict

    @classmethod
    def from_dict(cls, data: Dict):
        """Create job from dictionary"""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)
        self.results_dir = self.jobs_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.jobs: Dict[str, Job] = {}
        self.running_tasks: Dict[str, threading.Thread] = {}
        
        # Load existing jobs on startup
        self._load_jobs()
    
    def _load_jobs(self):
        """Load jobs from disk on startup"""
        jobs_file = self.jobs_dir / "jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                    for job_data in jobs_data:
                        job = Job.from_dict(job_data)
                        # Reset running jobs to pending on startup
                        if job.status == JobStatus.RUNNING:
                            job.status = JobStatus.PENDING
                        self.jobs[job.job_id] = job
            except Exception as e:
                print(f"Warning: Failed to load jobs from disk: {e}")
    
    def _save_jobs(self):
        """Save jobs to disk"""
        jobs_file = self.jobs_dir / "jobs.json"
        try:
            jobs_data = [job.to_dict() for job in self.jobs.values()]
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save jobs to disk: {e}")
    
    def create_job(self, task_type: str, request_params: Dict = None) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            task_type=task_type,
            created_at=datetime.now(timezone.utc),
            request_params=request_params
        )
        
        self.jobs[job_id] = job
        self._save_jobs()
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs"""
        return list(self.jobs.values())
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         progress: int = None, message: str = None, 
                         error_message: str = None):
        """Update job status and progress"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        old_status = job.status
        job.status = status
        
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.current_message = message
        if error_message is not None:
            job.error_message = error_message
        
        # Set timestamps
        now = datetime.now(timezone.utc)
        if old_status == JobStatus.PENDING and status == JobStatus.RUNNING:
            job.started_at = now
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = now
            # Clean up running task
            if job_id in self.running_tasks:
                del self.running_tasks[job_id]
        
        self._save_jobs()
        return True
    
    def set_job_result(self, job_id: str, result_data: Dict):
        """Save job results to file"""
        if job_id not in self.jobs:
            return None
        
        result_file = self.results_dir / f"{job_id}.json"
        try:
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.jobs[job_id].result_file = str(result_file)
            self._save_jobs()
            return str(result_file)
        except Exception as e:
            print(f"Failed to save job result: {e}")
            return None
    
    def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Get job results from file"""
        job = self.jobs.get(job_id)
        if not job or not job.result_file:
            return None
        
        try:
            with open(job.result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load job result: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False  # Already finished
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        
        # Stop the running task if it exists
        if job_id in self.running_tasks:
            # Note: This is a graceful cancellation - the thread will check status
            pass
        
        self._save_jobs()
        return True
    
    def start_background_task(self, job_id: str, task_func, *args, **kwargs):
        """Start a background task for a job"""
        if job_id not in self.jobs:
            return False
        
        def run_task():
            try:
                self.update_job_status(job_id, JobStatus.RUNNING, 0, "Starting task...")
                task_func(job_id, *args, **kwargs)
            except Exception as e:
                self.update_job_status(job_id, JobStatus.FAILED, 
                                     error_message=str(e))
                print(f"Task {job_id} failed: {e}")
        
        thread = threading.Thread(target=run_task, daemon=True)
        self.running_tasks[job_id] = thread
        thread.start()
        return True
    
    def cleanup_old_jobs(self, max_age_days: int = 7):
        """Clean up old completed jobs"""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 24 * 60 * 60)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and 
                job.completed_at and job.completed_at.timestamp() < cutoff):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            job = self.jobs.pop(job_id)
            # Clean up result file
            if job.result_file and Path(job.result_file).exists():
                try:
                    Path(job.result_file).unlink()
                except Exception:
                    pass
        
        if jobs_to_remove:
            self._save_jobs()
        
        return len(jobs_to_remove)