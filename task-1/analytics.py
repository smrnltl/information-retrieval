import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import os
from dataclasses import dataclass, asdict
import uuid

@dataclass
class SearchEvent:
    id: str
    timestamp: datetime
    query: str
    query_type: str
    user_id: Optional[str]
    results_count: int
    response_time: float
    filters_used: Dict
    clicked_results: List[int] = None

@dataclass
class ClickEvent:
    id: str
    timestamp: datetime
    search_id: str
    publication_id: int
    position: int
    user_id: Optional[str]

class AnalyticsDatabase:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                query_type TEXT,
                user_id TEXT,
                results_count INTEGER,
                response_time REAL,
                filters_used TEXT,
                clicked_results TEXT
            )
        ''')
        
        # Click events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS click_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                search_id TEXT,
                publication_id INTEGER,
                position INTEGER,
                user_id TEXT,
                FOREIGN KEY (search_id) REFERENCES search_events (id)
            )
        ''')
        
        # Popular queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS popular_queries (
                query TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1,
                last_searched TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                date TEXT PRIMARY KEY,
                avg_response_time REAL,
                total_searches INTEGER,
                unique_users INTEGER,
                popular_query TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

class AnalyticsTracker:
    def __init__(self, db_path: str = "analytics.db"):
        self.db = AnalyticsDatabase(db_path)
    
    def track_search(self, query: str, query_type: str, results_count: int, 
                    response_time: float, filters_used: Dict = None, 
                    user_id: str = None) -> str:
        """Track a search event"""
        search_id = str(uuid.uuid4())
        
        search_event = SearchEvent(
            id=search_id,
            timestamp=datetime.now(),
            query=query,
            query_type=query_type,
            user_id=user_id,
            results_count=results_count,
            response_time=response_time,
            filters_used=filters_used or {}
        )
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_events 
            (id, timestamp, query, query_type, user_id, results_count, response_time, filters_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            search_event.id,
            search_event.timestamp.isoformat(),
            search_event.query,
            search_event.query_type,
            search_event.user_id,
            search_event.results_count,
            search_event.response_time,
            json.dumps(search_event.filters_used)
        ))
        
        # Update popular queries
        cursor.execute('''
            INSERT OR REPLACE INTO popular_queries (query, count, last_searched)
            VALUES (?, COALESCE((SELECT count FROM popular_queries WHERE query = ?) + 1, 1), ?)
        ''', (query, query, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return search_id
    
    def track_click(self, search_id: str, publication_id: int, position: int, 
                   user_id: str = None) -> str:
        """Track a click event"""
        click_id = str(uuid.uuid4())
        
        click_event = ClickEvent(
            id=click_id,
            timestamp=datetime.now(),
            search_id=search_id,
            publication_id=publication_id,
            position=position,
            user_id=user_id
        )
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO click_events 
            (id, timestamp, search_id, publication_id, position, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            click_event.id,
            click_event.timestamp.isoformat(),
            click_event.search_id,
            click_event.publication_id,
            click_event.position,
            click_event.user_id
        ))
        
        conn.commit()
        conn.close()
        
        return click_id

class AnalyticsReporter:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
    
    def get_search_stats(self, days: int = 30) -> Dict:
        """Get search statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total searches
        cursor.execute('''
            SELECT COUNT(*) FROM search_events 
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        total_searches = cursor.fetchone()[0]
        
        # Unique users
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM search_events 
            WHERE timestamp >= ? AND user_id IS NOT NULL
        ''', (start_date.isoformat(),))
        unique_users = cursor.fetchone()[0]
        
        # Average response time
        cursor.execute('''
            SELECT AVG(response_time) FROM search_events 
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        avg_response_time = cursor.fetchone()[0] or 0
        
        # Most popular queries
        cursor.execute('''
            SELECT query, COUNT(*) as count FROM search_events 
            WHERE timestamp >= ?
            GROUP BY query 
            ORDER BY count DESC 
            LIMIT 10
        ''', (start_date.isoformat(),))
        popular_queries = cursor.fetchall()
        
        # Search trends by day
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM search_events 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp) 
            ORDER BY date
        ''', (start_date.isoformat(),))
        daily_searches = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_searches': total_searches,
            'unique_users': unique_users,
            'avg_response_time': round(avg_response_time, 3),
            'popular_queries': [{'query': q[0], 'count': q[1]} for q in popular_queries],
            'daily_searches': [{'date': d[0], 'count': d[1]} for d in daily_searches]
        }
    
    def get_click_through_rates(self, days: int = 30) -> Dict:
        """Calculate click-through rates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total searches with results
        cursor.execute('''
            SELECT COUNT(*) FROM search_events 
            WHERE timestamp >= ? AND results_count > 0
        ''', (start_date.isoformat(),))
        searches_with_results = cursor.fetchone()[0]
        
        # Searches that led to clicks
        cursor.execute('''
            SELECT COUNT(DISTINCT s.id) 
            FROM search_events s
            JOIN click_events c ON s.id = c.search_id
            WHERE s.timestamp >= ?
        ''', (start_date.isoformat(),))
        searches_with_clicks = cursor.fetchone()[0]
        
        # Click-through rate by position
        cursor.execute('''
            SELECT c.position, COUNT(*) as clicks
            FROM click_events c
            JOIN search_events s ON c.search_id = s.id
            WHERE s.timestamp >= ?
            GROUP BY c.position
            ORDER BY c.position
        ''', (start_date.isoformat(),))
        position_clicks = cursor.fetchall()
        
        # Most clicked publications
        cursor.execute('''
            SELECT c.publication_id, COUNT(*) as clicks
            FROM click_events c
            JOIN search_events s ON c.search_id = s.id
            WHERE s.timestamp >= ?
            GROUP BY c.publication_id
            ORDER BY clicks DESC
            LIMIT 10
        ''', (start_date.isoformat(),))
        popular_publications = cursor.fetchall()
        
        conn.close()
        
        ctr = (searches_with_clicks / searches_with_results * 100) if searches_with_results > 0 else 0
        
        return {
            'overall_ctr': round(ctr, 2),
            'searches_with_results': searches_with_results,
            'searches_with_clicks': searches_with_clicks,
            'position_clicks': [{'position': p[0], 'clicks': p[1]} for p in position_clicks],
            'popular_publications': [{'publication_id': p[0], 'clicks': p[1]} for p in popular_publications]
        }
    
    def get_query_analysis(self, days: int = 30) -> Dict:
        """Analyze query patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query types distribution
        cursor.execute('''
            SELECT query_type, COUNT(*) as count
            FROM search_events 
            WHERE timestamp >= ?
            GROUP BY query_type
        ''', (start_date.isoformat(),))
        query_types = cursor.fetchall()
        
        # Average query length
        cursor.execute('''
            SELECT AVG(LENGTH(query)) as avg_length
            FROM search_events 
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        avg_query_length = cursor.fetchone()[0] or 0
        
        # Queries with no results
        cursor.execute('''
            SELECT COUNT(*) FROM search_events 
            WHERE timestamp >= ? AND results_count = 0
        ''', (start_date.isoformat(),))
        no_results_count = cursor.fetchone()[0]
        
        # Most used filters
        cursor.execute('''
            SELECT filters_used FROM search_events 
            WHERE timestamp >= ? AND filters_used != '{}'
        ''', (start_date.isoformat(),))
        filters_data = cursor.fetchall()
        
        # Analyze filter usage
        filter_usage = defaultdict(int)
        for row in filters_data:
            try:
                filters = json.loads(row[0])
                for key in filters:
                    if filters[key]:  # Only count non-empty filters
                        filter_usage[key] += 1
            except:
                continue
        
        conn.close()
        
        return {
            'query_types': [{'type': t[0], 'count': t[1]} for t in query_types],
            'avg_query_length': round(avg_query_length, 1),
            'no_results_count': no_results_count,
            'filter_usage': dict(filter_usage)
        }
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """Get performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Response time percentiles
        cursor.execute('''
            SELECT response_time FROM search_events 
            WHERE timestamp >= ?
            ORDER BY response_time
        ''', (start_date.isoformat(),))
        
        response_times = [row[0] for row in cursor.fetchall() if row[0] is not None]
        
        if response_times:
            n = len(response_times)
            p50 = response_times[int(n * 0.5)] if n > 0 else 0
            p90 = response_times[int(n * 0.9)] if n > 0 else 0
            p95 = response_times[int(n * 0.95)] if n > 0 else 0
            p99 = response_times[int(n * 0.99)] if n > 0 else 0
        else:
            p50 = p90 = p95 = p99 = 0
        
        # Error rate (searches with 0 results)
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN results_count = 0 THEN 1 END) * 100.0 / COUNT(*) as error_rate
            FROM search_events 
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        error_rate = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'response_time_percentiles': {
                'p50': round(p50, 3),
                'p90': round(p90, 3),
                'p95': round(p95, 3),
                'p99': round(p99, 3)
            },
            'error_rate': round(error_rate, 2)
        }
    
    def generate_daily_report(self, date: str = None) -> Dict:
        """Generate daily analytics report"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        search_stats = self.get_search_stats(days=1)
        click_stats = self.get_click_through_rates(days=1)
        query_stats = self.get_query_analysis(days=1)
        performance_stats = self.get_performance_metrics(days=1)
        
        return {
            'date': date,
            'search_stats': search_stats,
            'click_stats': click_stats,
            'query_stats': query_stats,
            'performance_stats': performance_stats
        }

class AnalyticsExporter:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
    
    def export_to_json(self, output_file: str, days: int = 30):
        """Export analytics data to JSON"""
        reporter = AnalyticsReporter(self.db_path)
        
        data = {
            'export_date': datetime.now().isoformat(),
            'period_days': days,
            'search_stats': reporter.get_search_stats(days),
            'click_stats': reporter.get_click_through_rates(days),
            'query_analysis': reporter.get_query_analysis(days),
            'performance_metrics': reporter.get_performance_metrics(days)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Analytics data exported to {output_file}")

if __name__ == "__main__":
    # Test analytics system
    tracker = AnalyticsTracker()
    reporter = AnalyticsReporter()
    
    # Test tracking
    search_id = tracker.track_search(
        query="machine learning",
        query_type="simple",
        results_count=15,
        response_time=0.234,
        filters_used={"year": "2023"}
    )
    
    tracker.track_click(search_id, publication_id=123, position=1)
    
    # Test reporting
    stats = reporter.get_search_stats(days=30)
    print("Search Statistics:", json.dumps(stats, indent=2))
    
    # Export data
    exporter = AnalyticsExporter()
    exporter.export_to_json("analytics_report.json", days=30)