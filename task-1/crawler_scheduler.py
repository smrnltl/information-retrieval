"""
Crawler Scheduler System
Automated scheduling and management of web crawling tasks
"""

import schedule
import time
import threading
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
import subprocess
import sys
from pathlib import Path

# Import your existing crawler
from crawler import main_corrected_crawl

class CrawlFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class CrawlStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class CrawlJob:
    id: str
    name: str
    frequency: CrawlFrequency
    enabled: bool
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: CrawlStatus
    max_pages: Optional[int]
    max_workers: int
    custom_schedule: Optional[str]  # Cron-like expression
    created_at: datetime
    updated_at: datetime
    run_count: int
    success_count: int
    failure_count: int

@dataclass
class CrawlResult:
    job_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: CrawlStatus
    publications_found: int
    publications_added: int
    publications_updated: int
    error_message: Optional[str]
    execution_time: Optional[float]

class CrawlerDatabase:
    """Manage crawler scheduling database"""
    
    def __init__(self, db_path: str = "crawler_schedule.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize scheduler database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Crawl jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                frequency TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                last_run TEXT,
                next_run TEXT,
                status TEXT DEFAULT 'pending',
                max_pages INTEGER,
                max_workers INTEGER DEFAULT 2,
                custom_schedule TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                run_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0
            )
        ''')
        
        # Crawl results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL,
                publications_found INTEGER DEFAULT 0,
                publications_added INTEGER DEFAULT 0,
                publications_updated INTEGER DEFAULT 0,
                error_message TEXT,
                execution_time REAL,
                FOREIGN KEY (job_id) REFERENCES crawl_jobs (id)
            )
        ''')
        
        # Crawl logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES crawl_jobs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_job(self, job: CrawlJob):
        """Save or update a crawl job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO crawl_jobs 
            (id, name, frequency, enabled, last_run, next_run, status, max_pages, 
             max_workers, custom_schedule, created_at, updated_at, run_count, 
             success_count, failure_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.id, job.name, job.frequency.value, job.enabled,
            job.last_run.isoformat() if job.last_run else None,
            job.next_run.isoformat() if job.next_run else None,
            job.status.value, job.max_pages, job.max_workers, job.custom_schedule,
            job.created_at.isoformat(), job.updated_at.isoformat(),
            job.run_count, job.success_count, job.failure_count
        ))
        
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM crawl_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_job(row)
        return None
    
    def get_all_jobs(self) -> List[CrawlJob]:
        """Get all crawl jobs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM crawl_jobs ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_job(row) for row in rows]
    
    def get_enabled_jobs(self) -> List[CrawlJob]:
        """Get all enabled crawl jobs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM crawl_jobs WHERE enabled = TRUE')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_job(row) for row in rows]
    
    def _row_to_job(self, row) -> CrawlJob:
        """Convert database row to CrawlJob object"""
        return CrawlJob(
            id=row[0],
            name=row[1],
            frequency=CrawlFrequency(row[2]),
            enabled=bool(row[3]),
            last_run=datetime.fromisoformat(row[4]) if row[4] else None,
            next_run=datetime.fromisoformat(row[5]) if row[5] else None,
            status=CrawlStatus(row[6]),
            max_pages=row[7],
            max_workers=row[8],
            custom_schedule=row[9],
            created_at=datetime.fromisoformat(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
            run_count=row[12],
            success_count=row[13],
            failure_count=row[14]
        )
    
    def save_result(self, result: CrawlResult):
        """Save crawl result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO crawl_results 
            (job_id, start_time, end_time, status, publications_found, 
             publications_added, publications_updated, error_message, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.job_id,
            result.start_time.isoformat(),
            result.end_time.isoformat() if result.end_time else None,
            result.status.value,
            result.publications_found,
            result.publications_added,
            result.publications_updated,
            result.error_message,
            result.execution_time
        ))
        
        conn.commit()
        conn.close()
    
    def get_job_results(self, job_id: str, limit: int = 50) -> List[CrawlResult]:
        """Get results for a specific job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM crawl_results WHERE job_id = ? 
            ORDER BY start_time DESC LIMIT ?
        ''', (job_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append(CrawlResult(
                job_id=row[1],
                start_time=datetime.fromisoformat(row[2]),
                end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                status=CrawlStatus(row[4]),
                publications_found=row[5],
                publications_added=row[6],
                publications_updated=row[7],
                error_message=row[8],
                execution_time=row[9]
            ))
        
        return results
    
    def log_message(self, job_id: str, level: str, message: str):
        """Log a message for a job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO crawl_logs (job_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        ''', (job_id, datetime.now().isoformat(), level, message))
        
        conn.commit()
        conn.close()

class CrawlerScheduler:
    """Main crawler scheduler class"""
    
    def __init__(self, db_path: str = "crawler_schedule.db"):
        self.db = CrawlerDatabase(db_path)
        self.running = False
        self.scheduler_thread = None
        self.active_crawls = {}  # job_id -> thread
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_job(self, name: str, frequency: CrawlFrequency, 
                   max_pages: Optional[int] = None, max_workers: int = 2,
                   custom_schedule: Optional[str] = None) -> str:
        """Create a new crawl job"""
        job_id = f"crawl_{int(time.time())}_{hash(name) % 1000}"
        
        job = CrawlJob(
            id=job_id,
            name=name,
            frequency=frequency,
            enabled=True,
            last_run=None,
            next_run=self._calculate_next_run(frequency, custom_schedule),
            status=CrawlStatus.PENDING,
            max_pages=max_pages,
            max_workers=max_workers,
            custom_schedule=custom_schedule,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            run_count=0,
            success_count=0,
            failure_count=0
        )
        
        self.db.save_job(job)
        self.logger.info(f"Created crawl job: {job_id} - {name}")
        return job_id
    
    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update an existing job"""
        job = self.db.get_job(job_id)
        if not job:
            return False
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        job.updated_at = datetime.now()
        
        # Recalculate next run if frequency changed
        if 'frequency' in kwargs or 'custom_schedule' in kwargs:
            job.next_run = self._calculate_next_run(job.frequency, job.custom_schedule)
        
        self.db.save_job(job)
        return True
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a job"""
        return self.update_job(job_id, enabled=True)
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a job"""
        return self.update_job(job_id, enabled=False)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        # Stop if running
        if job_id in self.active_crawls:
            self.stop_job(job_id)
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM crawl_jobs WHERE id = ?', (job_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            self.logger.info(f"Deleted crawl job: {job_id}")
        
        return deleted
    
    def start_scheduler(self):
        """Start the scheduler"""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Crawler scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # Stop all active crawls
        for job_id in list(self.active_crawls.keys()):
            self.stop_job(job_id)
        
        self.logger.info("Crawler scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                jobs = self.db.get_enabled_jobs()
                current_time = datetime.now()
                
                for job in jobs:
                    # Skip if already running
                    if job.id in self.active_crawls:
                        continue
                    
                    # Check if it's time to run
                    if job.next_run and current_time >= job.next_run:
                        self._execute_job(job)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)
    
    def _execute_job(self, job: CrawlJob):
        """Execute a crawl job"""
        self.logger.info(f"Starting crawl job: {job.id} - {job.name}")
        
        # Update job status
        job.status = CrawlStatus.RUNNING
        job.last_run = datetime.now()
        job.next_run = self._calculate_next_run(job.frequency, job.custom_schedule)
        job.run_count += 1
        self.db.save_job(job)
        
        # Start crawl in separate thread
        crawl_thread = threading.Thread(
            target=self._run_crawler,
            args=(job,),
            daemon=True
        )
        
        self.active_crawls[job.id] = crawl_thread
        crawl_thread.start()
    
    def _run_crawler(self, job: CrawlJob):
        """Run the actual crawler"""
        start_time = datetime.now()
        result = CrawlResult(
            job_id=job.id,
            start_time=start_time,
            end_time=None,
            status=CrawlStatus.RUNNING,
            publications_found=0,
            publications_added=0,
            publications_updated=0,
            error_message=None,
            execution_time=None
        )
        
        try:
            self.db.log_message(job.id, "INFO", f"Starting crawl with max_pages={job.max_pages}, max_workers={job.max_workers}")
            
            # Count publications before crawling
            publications_before = self._count_publications()
            
            # Run the crawler
            success = main_corrected_crawl(
                headless=True,
                max_pages=job.max_pages,
                max_workers=job.max_workers
            )
            
            # Count publications after crawling
            publications_after = self._count_publications()
            
            # Calculate results
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            result.publications_found = publications_after
            result.publications_added = max(0, publications_after - publications_before)
            
            if success:
                result.status = CrawlStatus.SUCCESS
                job.status = CrawlStatus.SUCCESS
                job.success_count += 1
                
                self.db.log_message(job.id, "INFO", 
                    f"Crawl completed successfully. Found {result.publications_added} new publications")
                
                # Rebuild search index after successful crawl
                self._rebuild_search_index(job.id)
                
            else:
                result.status = CrawlStatus.FAILED
                job.status = CrawlStatus.FAILED
                job.failure_count += 1
                result.error_message = "Crawler returned failure status"
                
                self.db.log_message(job.id, "ERROR", "Crawl failed")
        
        except Exception as e:
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            result.status = CrawlStatus.FAILED
            result.error_message = str(e)
            
            job.status = CrawlStatus.FAILED
            job.failure_count += 1
            
            self.logger.error(f"Crawl job {job.id} failed: {e}")
            self.db.log_message(job.id, "ERROR", f"Crawl failed: {e}")
        
        finally:
            # Clean up
            if job.id in self.active_crawls:
                del self.active_crawls[job.id]
            
            # Save results
            self.db.save_job(job)
            self.db.save_result(result)
            
            self.logger.info(f"Crawl job {job.id} completed with status: {result.status.value}")
    
    def _count_publications(self) -> int:
        """Count current publications in database"""
        try:
            conn = sqlite3.connect("publications.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM publications")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Error counting publications: {e}")
            return 0
    
    def _rebuild_search_index(self, job_id: str):
        """Rebuild search index after successful crawl"""
        try:
            from search_engine import rebuild_index
            rebuild_index()
            self.db.log_message(job_id, "INFO", "Search index rebuilt successfully")
        except Exception as e:
            self.logger.error(f"Failed to rebuild search index: {e}")
            self.db.log_message(job_id, "ERROR", f"Failed to rebuild search index: {e}")
    
    def _calculate_next_run(self, frequency: CrawlFrequency, 
                           custom_schedule: Optional[str] = None) -> datetime:
        """Calculate next run time based on frequency"""
        now = datetime.now()
        
        if frequency == CrawlFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == CrawlFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == CrawlFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == CrawlFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == CrawlFrequency.CUSTOM and custom_schedule:
            # Simple custom schedule parsing (extend as needed)
            # Format: "daily@14:30" or "weekly@monday@09:00"
            return self._parse_custom_schedule(custom_schedule, now)
        else:
            return now + timedelta(hours=1)  # Default to hourly
    
    def _parse_custom_schedule(self, schedule: str, base_time: datetime) -> datetime:
        """Parse custom schedule string"""
        # Basic implementation - extend for more complex schedules
        if schedule.startswith("daily@"):
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_run = base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= base_time:
                next_run += timedelta(days=1)
            return next_run
        
        # Default fallback
        return base_time + timedelta(hours=1)
    
    def run_job_now(self, job_id: str) -> bool:
        """Run a job immediately"""
        job = self.db.get_job(job_id)
        if not job:
            return False
        
        if job.id in self.active_crawls:
            self.logger.warning(f"Job {job_id} is already running")
            return False
        
        self._execute_job(job)
        return True
    
    def stop_job(self, job_id: str) -> bool:
        """Stop a running job"""
        if job_id not in self.active_crawls:
            return False
        
        # This is a simple approach - for more complex stopping,
        # you'd need to implement proper thread communication
        try:
            thread = self.active_crawls[job_id]
            # Note: Python threads can't be forcefully stopped
            # In a production system, you'd use proper cancellation tokens
            self.logger.warning(f"Attempted to stop job {job_id} - thread will complete naturally")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get comprehensive job status"""
        job = self.db.get_job(job_id)
        if not job:
            return {}
        
        is_running = job_id in self.active_crawls
        recent_results = self.db.get_job_results(job_id, limit=5)
        
        return {
            "job": asdict(job),
            "is_running": is_running,
            "recent_results": [asdict(result) for result in recent_results],
            "success_rate": job.success_count / max(job.run_count, 1) * 100
        }
    
    def get_scheduler_status(self) -> Dict:
        """Get overall scheduler status"""
        jobs = self.db.get_all_jobs()
        
        return {
            "running": self.running,
            "total_jobs": len(jobs),
            "enabled_jobs": len([j for j in jobs if j.enabled]),
            "active_crawls": len(self.active_crawls),
            "active_job_ids": list(self.active_crawls.keys())
        }

def create_default_jobs():
    """Create some default crawl jobs"""
    scheduler = CrawlerScheduler()
    
    # Daily full crawl
    job_id1 = scheduler.create_job(
        name="Daily Full Crawl",
        frequency=CrawlFrequency.DAILY,
        max_pages=None,  # All pages
        max_workers=2
    )
    
    # Weekly comprehensive crawl
    job_id2 = scheduler.create_job(
        name="Weekly Comprehensive Crawl", 
        frequency=CrawlFrequency.WEEKLY,
        max_pages=None,
        max_workers=3
    )
    
    # Hourly quick check (limited pages)
    job_id3 = scheduler.create_job(
        name="Hourly Quick Check",
        frequency=CrawlFrequency.HOURLY,
        max_pages=2,  # Just check first 2 pages
        max_workers=1
    )
    
    print(f"Created default jobs:")
    print(f"  Daily: {job_id1}")
    print(f"  Weekly: {job_id2}")  
    print(f"  Hourly: {job_id3}")
    
    return scheduler

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawler Scheduler")
    parser.add_argument("--create-defaults", action="store_true", 
                       help="Create default crawl jobs")
    parser.add_argument("--start", action="store_true", 
                       help="Start the scheduler")
    parser.add_argument("--status", action="store_true", 
                       help="Show scheduler status")
    
    args = parser.parse_args()
    
    if args.create_defaults:
        create_default_jobs()
    
    if args.start:
        scheduler = CrawlerScheduler()
        scheduler.start_scheduler()
        print("Scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_scheduler()
    
    if args.status:
        scheduler = CrawlerScheduler()
        status = scheduler.get_scheduler_status()
        print(f"Scheduler Status: {json.dumps(status, indent=2)}")
        
        jobs = scheduler.db.get_all_jobs()
        print(f"\nJobs ({len(jobs)}):")
        for job in jobs:
            print(f"  {job.id}: {job.name} ({job.status.value}) - {job.frequency.value}")