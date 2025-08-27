"""
Crawler Configuration Management
Centralized configuration for all crawler components
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import sqlite3
from datetime import datetime

class CrawlerMode(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    QUICK = "quick"

class CrawlerStrategy(Enum):
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    PRIORITY_BASED = "priority_based"

@dataclass
class CrawlerSettings:
    """Core crawler settings"""
    # Basic settings
    base_url: str = "https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/"
    max_pages: Optional[int] = None
    max_workers: int = 2
    headless: bool = True
    
    # Timing and delays
    page_delay: float = 2.0
    worker_delay: float = 2.0
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 5.0
    
    # Browser settings
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    window_size: str = "1366,768"
    
    # Crawling behavior
    mode: CrawlerMode = CrawlerMode.FULL
    strategy: CrawlerStrategy = CrawlerStrategy.BREADTH_FIRST
    respect_robots_txt: bool = True
    follow_redirects: bool = True
    robots_user_agent: str = "Academic-Crawler/1.0 (Coventry University Research)"
    robots_cache_hours: int = 24
    
    # Data extraction
    extract_images: bool = False
    extract_pdfs: bool = False
    extract_citations: bool = True
    
    # Storage settings
    database_path: str = "publications.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Logging and monitoring
    log_level: str = "INFO"
    log_file: Optional[str] = "crawler.log"
    enable_metrics: bool = True
    
    # Rate limiting
    requests_per_minute: int = 30
    concurrent_requests: int = 2
    
    # Error handling
    max_consecutive_errors: int = 5
    error_cooldown_minutes: int = 10
    skip_on_error: bool = True
    
    # Advanced settings
    custom_selectors: Dict[str, str] = field(default_factory=lambda: {
        "publication_container": ".result-container",
        "title": "h3.title a.link span",
        "title_link": "h3.title a.link",
        "authors": "p.relations.persons",
        "abstract": "div.textblock",
        "date": "span.date",
        "pagination": "a.step"
    })
    
    # Site-specific settings
    cookie_settings: Dict[str, Any] = field(default_factory=lambda: {
        "accept_cookies": True,
        "cookie_button_selector": "#onetrust-accept-btn-handler"
    })
    
    # Performance tuning
    memory_limit_mb: int = 1024
    cache_size_mb: int = 100
    
    # Quality control
    min_title_length: int = 10
    min_abstract_length: int = 50
    validate_urls: bool = True
    deduplicate: bool = True

@dataclass
class SchedulerSettings:
    """Scheduler-specific settings"""
    enabled: bool = True
    check_interval_seconds: int = 60
    max_concurrent_jobs: int = 3
    job_timeout_hours: int = 6
    cleanup_old_jobs_days: int = 30
    notification_enabled: bool = False
    notification_webhook: Optional[str] = None

@dataclass
class APISettings:
    """API server settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    auth_enabled: bool = False
    api_key: Optional[str] = None

@dataclass
class DatabaseSettings:
    """Database configuration"""
    type: str = "sqlite"
    path: str = "publications.db"
    backup_enabled: bool = True
    backup_path: str = "backups/"
    auto_vacuum: bool = True
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    
    # For future PostgreSQL support
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None

@dataclass
class SystemConfiguration:
    """Complete system configuration"""
    crawler: CrawlerSettings = field(default_factory=CrawlerSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    api: APISettings = field(default_factory=APISettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    
    # System metadata
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ConfigurationManager:
    """Manage system configuration"""
    
    def __init__(self, config_path: str = "crawler_config.json"):
        self.config_path = config_path
        self.config = SystemConfiguration()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Parse configuration sections
                if 'crawler' in data:
                    self.config.crawler = CrawlerSettings(**data['crawler'])
                if 'scheduler' in data:
                    self.config.scheduler = SchedulerSettings(**data['scheduler'])
                if 'api' in data:
                    self.config.api = APISettings(**data['api'])
                if 'database' in data:
                    self.config.database = DatabaseSettings(**data['database'])
                
                # System metadata
                self.config.version = data.get('version', '1.0.0')
                if 'created_at' in data:
                    self.config.created_at = datetime.fromisoformat(data['created_at'])
                if 'updated_at' in data:
                    self.config.updated_at = datetime.fromisoformat(data['updated_at'])
                
                print(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                print(f"Error loading configuration: {e}")
                print("Using default configuration")
        else:
            print(f"Configuration file {self.config_path} not found, using defaults")
            self.save_config()  # Save defaults
    
    def save_config(self):
        """Save configuration to file"""
        try:
            # Update timestamp
            self.config.updated_at = datetime.now()
            if not self.config.created_at:
                self.config.created_at = datetime.now()
            
            # Convert to dict
            config_dict = {
                'crawler': asdict(self.config.crawler),
                'scheduler': asdict(self.config.scheduler),
                'api': asdict(self.config.api),
                'database': asdict(self.config.database),
                'version': self.config.version,
                'created_at': self.config.created_at.isoformat(),
                'updated_at': self.config.updated_at.isoformat()
            }
            
            # Handle enums
            config_dict['crawler']['mode'] = config_dict['crawler']['mode'].value
            config_dict['crawler']['strategy'] = config_dict['crawler']['strategy'].value
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_crawler_settings(self) -> CrawlerSettings:
        """Get crawler settings"""
        return self.config.crawler
    
    def get_scheduler_settings(self) -> SchedulerSettings:
        """Get scheduler settings"""
        return self.config.scheduler
    
    def get_api_settings(self) -> APISettings:
        """Get API settings"""
        return self.config.api
    
    def get_database_settings(self) -> DatabaseSettings:
        """Get database settings"""
        return self.config.database
    
    def update_crawler_settings(self, **kwargs):
        """Update crawler settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.crawler, key):
                setattr(self.config.crawler, key, value)
        self.save_config()
    
    def update_scheduler_settings(self, **kwargs):
        """Update scheduler settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.scheduler, key):
                setattr(self.config.scheduler, key, value)
        self.save_config()
    
    def update_api_settings(self, **kwargs):
        """Update API settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.api, key):
                setattr(self.config.api, key, value)
        self.save_config()
    
    def update_database_settings(self, **kwargs):
        """Update database settings"""
        for key, value in kwargs.items():
            if hasattr(self.config.database, key):
                setattr(self.config.database, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = SystemConfiguration()
        self.save_config()
        print("Configuration reset to defaults")
    
    def export_config(self, export_path: str):
        """Export configuration to specified path"""
        try:
            import shutil
            shutil.copy(self.config_path, export_path)
            print(f"Configuration exported to {export_path}")
        except Exception as e:
            print(f"Error exporting configuration: {e}")
    
    def import_config(self, import_path: str):
        """Import configuration from specified path"""
        try:
            import shutil
            shutil.copy(import_path, self.config_path)
            self.load_config()
            print(f"Configuration imported from {import_path}")
        except Exception as e:
            print(f"Error importing configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate crawler settings
        if self.config.crawler.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if self.config.crawler.max_workers > 10:
            errors.append("max_workers should not exceed 10 for stability")
        
        if self.config.crawler.page_delay < 0:
            errors.append("page_delay cannot be negative")
        
        if self.config.crawler.requests_per_minute < 1:
            errors.append("requests_per_minute must be at least 1")
        
        # Validate paths
        if not self.config.crawler.base_url.startswith(('http://', 'https://')):
            errors.append("base_url must be a valid URL")
        
        # Validate scheduler settings
        if self.config.scheduler.check_interval_seconds < 30:
            errors.append("check_interval_seconds should be at least 30 seconds")
        
        if self.config.scheduler.max_concurrent_jobs > 10:
            errors.append("max_concurrent_jobs should not exceed 10")
        
        # Validate API settings
        if not (1024 <= self.config.api.port <= 65535):
            errors.append("API port must be between 1024 and 65535")
        
        if self.config.api.rate_limit_requests < 1:
            errors.append("rate_limit_requests must be at least 1")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "version": self.config.version,
            "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
            "updated_at": self.config.updated_at.isoformat() if self.config.updated_at else None,
            "crawler": {
                "mode": self.config.crawler.mode.value,
                "max_pages": self.config.crawler.max_pages,
                "max_workers": self.config.crawler.max_workers,
                "headless": self.config.crawler.headless
            },
            "scheduler": {
                "enabled": self.config.scheduler.enabled,
                "max_concurrent_jobs": self.config.scheduler.max_concurrent_jobs
            },
            "api": {
                "port": self.config.api.port,
                "cors_enabled": self.config.api.cors_enabled
            },
            "database": {
                "type": self.config.database.type,
                "path": self.config.database.path
            }
        }

class ConfigurationDatabase:
    """Store configuration in database for web interface"""
    
    def __init__(self, db_path: str = "config.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize configuration database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(section, key)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_at TEXT NOT NULL,
                changed_by TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_configuration(self, config: SystemConfiguration):
        """Save configuration to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Save crawler settings
        crawler_dict = asdict(config.crawler)
        for key, value in crawler_dict.items():
            cursor.execute('''
                INSERT OR REPLACE INTO configuration (section, key, value, type, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', ('crawler', key, json.dumps(value), type(value).__name__, now))
        
        # Save other sections similarly...
        
        conn.commit()
        conn.close()

# Global configuration instance
config_manager = ConfigurationManager()

def get_config() -> SystemConfiguration:
    """Get global configuration"""
    return config_manager.config

def get_crawler_config() -> CrawlerSettings:
    """Get crawler configuration"""
    return config_manager.get_crawler_settings()

def get_scheduler_config() -> SchedulerSettings:
    """Get scheduler configuration"""
    return config_manager.get_scheduler_settings()

def get_api_config() -> APISettings:
    """Get API configuration"""
    return config_manager.get_api_settings()

def get_database_config() -> DatabaseSettings:
    """Get database configuration"""
    return config_manager.get_database_settings()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawler Configuration Manager")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--reset", action="store_true", help="Reset to defaults")
    parser.add_argument("--export", type=str, help="Export config to file")
    parser.add_argument("--import", type=str, dest="import_file", help="Import config from file")
    
    args = parser.parse_args()
    
    if args.show:
        summary = config_manager.get_config_summary()
        print("Current Configuration:")
        print(json.dumps(summary, indent=2))
    
    elif args.validate:
        errors = config_manager.validate_config()
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Configuration is valid")
    
    elif args.reset:
        config_manager.reset_to_defaults()
        print("Configuration reset to defaults")
    
    elif args.export:
        config_manager.export_config(args.export)
    
    elif args.import_file:
        config_manager.import_config(args.import_file)
    
    else:
        print("Use --help for available options")