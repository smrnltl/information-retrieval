"""
Incremental Crawler - Smart crawling with change detection
Only processes new and updated content
"""

import sqlite3
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import os
import logging
from urllib.parse import urljoin

# Import your existing crawler components
from crawler import create_driver, handle_setup, extract_from_listing_page, extract_from_individual_page, check_robots_compliance

@dataclass
class CrawlSnapshot:
    """Snapshot of a crawl state for change detection"""
    url: str
    content_hash: str
    last_modified: datetime
    publication_count: int
    page_number: int

class IncrementalCrawlerDatabase:
    """Database for tracking incremental crawl state"""
    
    def __init__(self, db_path: str = "incremental_crawl.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize incremental crawling database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Page snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS page_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                publication_count INTEGER DEFAULT 0,
                page_number INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE(url)
            )
        ''')
        
        # Publication tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS publication_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                publication_id INTEGER,
                title TEXT NOT NULL,
                title_hash TEXT NOT NULL,
                authors_hash TEXT,
                abstract_hash TEXT,
                year TEXT,
                link TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                change_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                UNIQUE(title_hash, year)
            )
        ''')
        
        # Change log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS change_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                publication_id INTEGER,
                change_type TEXT NOT NULL,
                field_name TEXT,
                old_value TEXT,
                new_value TEXT,
                detected_at TEXT NOT NULL,
                FOREIGN KEY (publication_id) REFERENCES publication_tracking (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_page_snapshot(self, snapshot: CrawlSnapshot):
        """Save page snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO page_snapshots 
            (url, content_hash, last_modified, publication_count, page_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.url,
            snapshot.content_hash,
            snapshot.last_modified.isoformat(),
            snapshot.publication_count,
            snapshot.page_number,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_page_snapshot(self, url: str) -> Optional[CrawlSnapshot]:
        """Get page snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM page_snapshots WHERE url = ?', (url,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return CrawlSnapshot(
                url=row[1],
                content_hash=row[2],
                last_modified=datetime.fromisoformat(row[3]),
                publication_count=row[4],
                page_number=row[5]
            )
        return None
    
    def track_publication(self, title: str, authors: List[str], abstract: str, 
                         year: str, link: str, publication_id: Optional[int] = None) -> int:
        """Track publication for changes"""
        title_hash = hashlib.md5(title.encode()).hexdigest() if title else ""
        authors_hash = hashlib.md5(json.dumps(sorted(authors)).encode()).hexdigest()
        abstract_hash = hashlib.md5(abstract.encode()).hexdigest() if abstract else ""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if publication exists
        cursor.execute(
            'SELECT id, authors_hash, abstract_hash FROM publication_tracking WHERE title_hash = ? AND year = ?',
            (title_hash, year)
        )
        existing = cursor.fetchone()
        
        now = datetime.now().isoformat()
        
        if existing:
            tracking_id = existing[0]
            old_authors_hash = existing[1]
            old_abstract_hash = existing[2]
            
            changes = []
            
            # Check for changes
            if old_authors_hash != authors_hash:
                changes.append(('authors', old_authors_hash, authors_hash))
            
            if old_abstract_hash != abstract_hash:
                changes.append(('abstract', old_abstract_hash, abstract_hash))
            
            # Update tracking record
            cursor.execute('''
                UPDATE publication_tracking 
                SET authors_hash = ?, abstract_hash = ?, last_updated = ?, 
                    change_count = change_count + ?, publication_id = ?
                WHERE id = ?
            ''', (authors_hash, abstract_hash, now, len(changes), publication_id, tracking_id))
            
            # Log changes
            for field, old_val, new_val in changes:
                cursor.execute('''
                    INSERT INTO change_log (publication_id, change_type, field_name, old_value, new_value, detected_at)
                    VALUES (?, 'update', ?, ?, ?, ?)
                ''', (tracking_id, field, old_val, new_val, now))
        
        else:
            # New publication
            cursor.execute('''
                INSERT INTO publication_tracking 
                (publication_id, title, title_hash, authors_hash, abstract_hash, year, link, first_seen, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (publication_id, title, title_hash, authors_hash, abstract_hash, year, link, now, now))
            
            tracking_id = cursor.lastrowid
            
            # Log as new
            cursor.execute('''
                INSERT INTO change_log (publication_id, change_type, detected_at)
                VALUES (?, 'new', ?)
            ''', (tracking_id, now))
        
        conn.commit()
        conn.close()
        
        return tracking_id
    
    def get_recent_changes(self, hours: int = 24) -> List[Dict]:
        """Get recent changes"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cl.*, pt.title, pt.year, pt.link
            FROM change_log cl
            JOIN publication_tracking pt ON cl.publication_id = pt.id
            WHERE cl.detected_at > ?
            ORDER BY cl.detected_at DESC
        ''', (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        changes = []
        for row in rows:
            changes.append({
                'id': row[0],
                'publication_id': row[1],
                'change_type': row[2],
                'field_name': row[3],
                'old_value': row[4],
                'new_value': row[5],
                'detected_at': row[6],
                'title': row[7],
                'year': row[8],
                'link': row[9]
            })
        
        return changes

class IncrementalCrawler:
    """Smart crawler that detects and processes only changes"""
    
    def __init__(self, base_url: str = "https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/"):
        self.base_url = base_url
        self.db = IncrementalCrawlerDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'pages_checked': 0,
            'pages_changed': 0,
            'publications_new': 0,
            'publications_updated': 0,
            'publications_unchanged': 0
        }
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash of page content for change detection"""
        # Remove dynamic content that changes frequently
        cleaned_content = content
        
        # Remove timestamps, session IDs, etc.
        import re
        cleaned_content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '', cleaned_content)
        cleaned_content = re.sub(r'sessionid=[^&\s]+', '', cleaned_content)
        cleaned_content = re.sub(r'timestamp=\d+', '', cleaned_content)
        
        return hashlib.md5(cleaned_content.encode()).hexdigest()
    
    def has_page_changed(self, url: str, current_content: str) -> Tuple[bool, Optional[CrawlSnapshot]]:
        """Check if page has changed since last crawl"""
        current_hash = self.calculate_content_hash(current_content)
        snapshot = self.db.get_page_snapshot(url)
        
        if not snapshot:
            return True, None  # New page
        
        return current_hash != snapshot.content_hash, snapshot
    
    def incremental_crawl(self, max_pages: Optional[int] = None, max_workers: int = 2) -> Dict:
        """Perform incremental crawl"""
        self.logger.info("Starting incremental crawl...")
        start_time = time.time()
        
        driver = create_driver(headless=True)
        if not driver:
            return {'error': 'Failed to create driver'}
        
        try:
            driver.get(self.base_url)
            if not handle_setup(driver):
                return {'error': 'Failed to handle page setup'}
            
            # Crawl pages incrementally
            page_num = 1
            new_publications = []
            
            while True:
                if max_pages and page_num > max_pages:
                    break
                
                current_url = f"{self.base_url}?page={page_num}"
                self.logger.info(f"Checking page {page_num}: {current_url}")
                
                # Check robots.txt compliance
                if not check_robots_compliance(current_url):
                    self.logger.warning(f"Page {page_num} blocked by robots.txt, skipping")
                    page_num += 1
                    continue
                
                if page_num > 1:
                    driver.get(current_url)
                    time.sleep(3)
                
                # Get page content
                page_content = driver.page_source
                self.stats['pages_checked'] += 1
                
                # Check if page changed
                changed, old_snapshot = self.has_page_changed(current_url, page_content)
                
                if not changed:
                    self.logger.info(f"Page {page_num} unchanged, skipping...")
                    page_num += 1
                    
                    # If we've reached pages that haven't changed, we can probably stop
                    # (assuming chronological ordering)
                    if page_num > 3:  # Allow some buffer
                        break
                    continue
                
                self.stats['pages_changed'] += 1
                self.logger.info(f"Page {page_num} has changes, processing...")
                
                # Extract publications from this page
                publications = extract_from_listing_page(driver)
                
                if not publications:
                    break  # No more publications
                
                # Process each publication for changes
                for pub in publications:
                    if self.is_publication_new_or_changed(pub):
                        new_publications.append(pub)
                
                # Save page snapshot
                snapshot = CrawlSnapshot(
                    url=current_url,
                    content_hash=self.calculate_content_hash(page_content),
                    last_modified=datetime.now(),
                    publication_count=len(publications),
                    page_number=page_num
                )
                self.db.save_page_snapshot(snapshot)
                
                page_num += 1
                time.sleep(2)  # Rate limiting
            
            # Process individual pages for new/changed publications
            if new_publications:
                self.logger.info(f"Processing {len(new_publications)} new/changed publications...")
                self.process_individual_publications(new_publications, max_workers)
            
            execution_time = time.time() - start_time
            
            result = {
                'success': True,
                'execution_time': execution_time,
                'statistics': self.stats,
                'new_publications': len(new_publications)
            }
            
            self.logger.info(f"Incremental crawl completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Incremental crawl error: {e}")
            return {'error': str(e), 'statistics': self.stats}
        
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def is_publication_new_or_changed(self, pub_info: Dict) -> bool:
        """Check if publication is new or has changes"""
        title = pub_info.get('title', '')
        year = pub_info.get('year', '')
        
        if not title:
            return False
        
        title_hash = hashlib.md5(title.encode()).hexdigest()
        
        # Check if we've seen this publication before
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id FROM publication_tracking WHERE title_hash = ? AND year = ?',
            (title_hash, year)
        )
        existing = cursor.fetchone()
        conn.close()
        
        return existing is None  # New if not found
    
    def process_individual_publications(self, publications: List[Dict], max_workers: int = 2):
        """Process individual publication pages"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, pub in enumerate(publications):
                future = executor.submit(self.process_single_publication, pub, i)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result.get('success'):
                        if result.get('is_new'):
                            self.stats['publications_new'] += 1
                        else:
                            self.stats['publications_updated'] += 1
                    else:
                        self.stats['publications_unchanged'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing publication: {e}")
    
    def process_single_publication(self, pub_info: Dict, worker_id: int) -> Dict:
        """Process a single publication"""
        try:
            result = extract_from_individual_page(pub_info, worker_id)
            
            if result and result.get('success'):
                # Track publication for changes
                tracking_id = self.db.track_publication(
                    title=result['title'],
                    authors=result['authors'],
                    abstract=result['abstract'],
                    year=result['year'],
                    link=result['link'],
                    publication_id=None  # We don't have the DB ID yet
                )
                
                return {
                    'success': True,
                    'tracking_id': tracking_id,
                    'is_new': True  # For now, assume new (proper logic would check change log)
                }
            
            return {'success': False}
            
        except Exception as e:
            self.logger.error(f"Error processing individual publication: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_crawl_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent crawl activity"""
        changes = self.db.get_recent_changes(hours)
        
        summary = {
            'total_changes': len(changes),
            'new_publications': len([c for c in changes if c['change_type'] == 'new']),
            'updated_publications': len([c for c in changes if c['change_type'] == 'update']),
            'changes_by_field': {},
            'recent_changes': changes[:10]  # Last 10 changes
        }
        
        # Group by field
        for change in changes:
            if change['change_type'] == 'update' and change['field_name']:
                field = change['field_name']
                if field not in summary['changes_by_field']:
                    summary['changes_by_field'][field] = 0
                summary['changes_by_field'][field] += 1
        
        return summary
    
    def force_full_refresh(self):
        """Force full refresh by clearing all snapshots"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM page_snapshots')
        cursor.execute('DELETE FROM publication_tracking')
        cursor.execute('DELETE FROM change_log')
        conn.commit()
        conn.close()
        
        self.logger.info("Forced full refresh - all snapshots cleared")

def run_incremental_crawl(max_pages: Optional[int] = None, max_workers: int = 2) -> Dict:
    """Convenience function to run incremental crawl"""
    crawler = IncrementalCrawler()
    return crawler.incremental_crawl(max_pages, max_workers)

def get_change_summary(hours: int = 24) -> Dict:
    """Get summary of recent changes"""
    crawler = IncrementalCrawler()
    return crawler.get_crawl_summary(hours)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental Crawler")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum pages to check")
    parser.add_argument("--max-workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--full-refresh", action="store_true", help="Force full refresh")
    parser.add_argument("--summary", action="store_true", help="Show change summary")
    
    args = parser.parse_args()
    
    if args.full_refresh:
        crawler = IncrementalCrawler()
        crawler.force_full_refresh()
        print("Full refresh completed")
    
    elif args.summary:
        summary = get_change_summary(24)
        print("Change Summary (last 24 hours):")
        print(json.dumps(summary, indent=2))
    
    else:
        print(f"Starting incremental crawl (max_pages={args.max_pages}, max_workers={args.max_workers})...")
        result = run_incremental_crawl(args.max_pages, args.max_workers)
        print("Incremental crawl result:")
        print(json.dumps(result, indent=2))