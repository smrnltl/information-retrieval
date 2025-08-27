"""
Robots.txt Parser and Compliance Module
Ensures web crawling respects robots.txt directives
"""

import requests
import time
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

@dataclass
class RobotRule:
    """Individual robot rule"""
    user_agent: str
    disallow: List[str]
    allow: List[str]
    crawl_delay: Optional[float] = None
    request_rate: Optional[str] = None

@dataclass 
class SitemapInfo:
    """Sitemap information from robots.txt"""
    url: str
    last_checked: Optional[datetime] = None

class RobotsParser:
    """Enhanced robots.txt parser with crawl delay and rate limiting"""
    
    def __init__(self, user_agent: str = "*", timeout: int = 30):
        self.user_agent = user_agent
        self.timeout = timeout
        self.robots_cache: Dict[str, Tuple[RobotFileParser, datetime]] = {}
        self.cache_duration = timedelta(hours=24)  # Cache robots.txt for 24 hours
        self.logger = logging.getLogger(__name__)
        
        # Track crawl delays per domain
        self.crawl_delays: Dict[str, float] = {}
        self.last_request_time: Dict[str, datetime] = {}
        
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL for a given URL"""
        domain = self._get_domain(url)
        return urljoin(domain, '/robots.txt')
    
    def fetch_and_parse_robots(self, url: str, force_refresh: bool = False) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt for a given URL"""
        domain = self._get_domain(url)
        robots_url = self._get_robots_url(url)
        
        # Check cache first
        if not force_refresh and domain in self.robots_cache:
            cached_parser, cached_time = self.robots_cache[domain]
            if datetime.now() - cached_time < self.cache_duration:
                self.logger.debug(f"Using cached robots.txt for {domain}")
                return cached_parser
        
        try:
            self.logger.info(f"Fetching robots.txt from {robots_url}")
            
            # Create RobotFileParser instance
            parser = RobotFileParser()
            parser.set_url(robots_url)
            
            # Fetch with timeout and proper headers
            response = requests.get(
                robots_url,
                timeout=self.timeout,
                headers={
                    'User-Agent': f'Academic-Crawler/1.0 ({self.user_agent})',
                    'Accept': 'text/plain, */*'
                }
            )
            
            if response.status_code == 200:
                # Parse the robots.txt content
                robots_content = response.text
                parser.set_url(robots_url)
                parser.read()  # This will read from the URL
                
                # Also parse manually for additional directives
                self._parse_additional_directives(robots_content, domain)
                
                # Cache the parser
                self.robots_cache[domain] = (parser, datetime.now())
                
                self.logger.info(f"Successfully parsed robots.txt for {domain}")
                return parser
                
            elif response.status_code == 404:
                self.logger.info(f"No robots.txt found for {domain} - allowing all")
                # Create permissive parser
                parser = RobotFileParser()
                parser.set_url(robots_url)
                self.robots_cache[domain] = (parser, datetime.now())
                return parser
                
            else:
                self.logger.warning(f"Unexpected status code {response.status_code} for robots.txt at {robots_url}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Error fetching robots.txt from {robots_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing robots.txt from {robots_url}: {e}")
            return None
    
    def _parse_additional_directives(self, robots_content: str, domain: str):
        """Parse additional directives like Crawl-delay and Request-rate"""
        lines = robots_content.split('\n')
        current_user_agent = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.lower().startswith('user-agent:'):
                current_user_agent = line.split(':', 1)[1].strip()
                continue
            
            # Check if this applies to our user agent
            if current_user_agent and not self._matches_user_agent(current_user_agent):
                continue
                
            # Parse Crawl-delay
            if line.lower().startswith('crawl-delay:'):
                try:
                    delay = float(line.split(':', 1)[1].strip())
                    self.crawl_delays[domain] = delay
                    self.logger.info(f"Found crawl-delay {delay}s for {domain}")
                except ValueError:
                    pass
            
            # Parse Request-rate (requests per time period)
            elif line.lower().startswith('request-rate:'):
                rate_str = line.split(':', 1)[1].strip()
                # Format: "1/10s" means 1 request per 10 seconds
                match = re.match(r'(\d+)/(\d+)([smh]?)', rate_str)
                if match:
                    requests_num = int(match.group(1))
                    time_num = int(match.group(2))
                    time_unit = match.group(3) or 's'
                    
                    # Convert to seconds
                    multiplier = {'s': 1, 'm': 60, 'h': 3600}.get(time_unit, 1)
                    delay = (time_num * multiplier) / requests_num
                    
                    self.crawl_delays[domain] = max(
                        self.crawl_delays.get(domain, 0), 
                        delay
                    )
                    self.logger.info(f"Found request-rate {rate_str}, setting delay to {delay}s for {domain}")
    
    def _matches_user_agent(self, robots_user_agent: str) -> bool:
        """Check if robots.txt user-agent matches our user agent"""
        robots_user_agent = robots_user_agent.lower()
        our_user_agent = self.user_agent.lower()
        
        if robots_user_agent == '*':
            return True
        if robots_user_agent == our_user_agent:
            return True
        if robots_user_agent in our_user_agent:
            return True
            
        return False
    
    def can_fetch(self, url: str, user_agent: Optional[str] = None) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if user_agent is None:
            user_agent = self.user_agent
            
        domain = self._get_domain(url)
        parser = self.fetch_and_parse_robots(url)
        
        if parser is None:
            # If we can't fetch robots.txt, be conservative and allow
            self.logger.warning(f"Could not fetch robots.txt for {domain}, allowing by default")
            return True
        
        try:
            can_fetch = parser.can_fetch(user_agent, url)
            self.logger.debug(f"robots.txt check for {url}: {'ALLOWED' if can_fetch else 'DISALLOWED'}")
            return can_fetch
        except Exception as e:
            self.logger.error(f"Error checking robots.txt for {url}: {e}")
            return True  # Allow by default if there's an error
    
    def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay for a domain"""
        domain = self._get_domain(url)
        
        # First check our parsed delays
        if domain in self.crawl_delays:
            return self.crawl_delays[domain]
        
        # Check robots.txt parser
        parser = self.fetch_and_parse_robots(url)
        if parser:
            try:
                delay = parser.crawl_delay(self.user_agent)
                if delay:
                    self.crawl_delays[domain] = float(delay)
                    return float(delay)
            except:
                pass
        
        # Default delay
        return 1.0
    
    def wait_if_needed(self, url: str):
        """Wait if needed based on crawl delay and last request time"""
        domain = self._get_domain(url)
        crawl_delay = self.get_crawl_delay(url)
        
        if domain in self.last_request_time:
            time_since_last = datetime.now() - self.last_request_time[domain]
            time_since_seconds = time_since_last.total_seconds()
            
            if time_since_seconds < crawl_delay:
                wait_time = crawl_delay - time_since_seconds
                self.logger.info(f"Waiting {wait_time:.2f}s for crawl delay compliance ({domain})")
                time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time[domain] = datetime.now()
    
    def get_sitemaps(self, url: str) -> List[str]:
        """Extract sitemap URLs from robots.txt"""
        domain = self._get_domain(url)
        robots_url = self._get_robots_url(url)
        sitemaps = []
        
        try:
            response = requests.get(robots_url, timeout=self.timeout)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemaps.append(sitemap_url)
                        
        except Exception as e:
            self.logger.error(f"Error fetching sitemaps from {robots_url}: {e}")
        
        return sitemaps
    
    def get_robots_summary(self, url: str) -> Dict:
        """Get summary of robots.txt rules for a URL"""
        domain = self._get_domain(url)
        parser = self.fetch_and_parse_robots(url)
        
        summary = {
            'domain': domain,
            'robots_url': self._get_robots_url(url),
            'can_fetch': self.can_fetch(url),
            'crawl_delay': self.get_crawl_delay(url),
            'sitemaps': self.get_sitemaps(url),
            'last_checked': datetime.now().isoformat()
        }
        
        if parser:
            try:
                # Get disallowed paths
                summary['disallowed_paths'] = []
                summary['allowed_paths'] = []
                
                # This is a simplified approach - would need more sophisticated parsing for full details
                robots_url = self._get_robots_url(url)
                response = requests.get(robots_url, timeout=self.timeout)
                if response.status_code == 200:
                    content = response.text
                    lines = content.split('\n')
                    current_user_agent = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        if line.lower().startswith('user-agent:'):
                            current_user_agent = line.split(':', 1)[1].strip()
                        elif line.lower().startswith('disallow:') and self._matches_user_agent(current_user_agent or '*'):
                            path = line.split(':', 1)[1].strip()
                            if path:
                                summary['disallowed_paths'].append(path)
                        elif line.lower().startswith('allow:') and self._matches_user_agent(current_user_agent or '*'):
                            path = line.split(':', 1)[1].strip()
                            if path:
                                summary['allowed_paths'].append(path)
                                
            except Exception as e:
                self.logger.error(f"Error getting robots summary: {e}")
        
        return summary

class RobotsCompliantCrawler:
    """Wrapper class to make any crawler robots.txt compliant"""
    
    def __init__(self, user_agent: str = "Academic-Crawler/1.0", respect_robots: bool = True):
        self.robots_parser = RobotsParser(user_agent)
        self.respect_robots = respect_robots
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'urls_checked': 0,
            'urls_allowed': 0,
            'urls_disallowed': 0,
            'total_delay_time': 0.0
        }
    
    def can_crawl(self, url: str) -> bool:
        """Check if URL can be crawled"""
        self.stats['urls_checked'] += 1
        
        if not self.respect_robots:
            self.stats['urls_allowed'] += 1
            return True
        
        allowed = self.robots_parser.can_fetch(url)
        if allowed:
            self.stats['urls_allowed'] += 1
        else:
            self.stats['urls_disallowed'] += 1
            self.logger.info(f"URL blocked by robots.txt: {url}")
        
        return allowed
    
    def pre_crawl_wait(self, url: str):
        """Wait before crawling if needed"""
        if not self.respect_robots:
            return
        
        start_time = time.time()
        self.robots_parser.wait_if_needed(url)
        wait_time = time.time() - start_time
        self.stats['total_delay_time'] += wait_time
    
    def get_crawl_stats(self) -> Dict:
        """Get crawling statistics"""
        return {
            **self.stats,
            'compliance_rate': self.stats['urls_allowed'] / max(self.stats['urls_checked'], 1) * 100,
            'avg_delay_per_request': self.stats['total_delay_time'] / max(self.stats['urls_checked'], 1)
        }
    
    def get_domain_summary(self, url: str) -> Dict:
        """Get robots.txt summary for domain"""
        return self.robots_parser.get_robots_summary(url)

# Global instance for easy access
default_robots_checker = RobotsCompliantCrawler()

def check_robots_compliance(url: str, user_agent: str = "Academic-Crawler/1.0") -> bool:
    """Quick function to check if URL is allowed by robots.txt"""
    parser = RobotsParser(user_agent)
    return parser.can_fetch(url)

def get_crawl_delay(url: str, user_agent: str = "Academic-Crawler/1.0") -> float:
    """Quick function to get crawl delay for URL"""
    parser = RobotsParser(user_agent) 
    return parser.get_crawl_delay(url)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robots.txt Compliance Checker")
    parser.add_argument("url", help="URL to check")
    parser.add_argument("--user-agent", default="Academic-Crawler/1.0", help="User agent string")
    parser.add_argument("--summary", action="store_true", help="Show robots.txt summary")
    parser.add_argument("--delay", action="store_true", help="Show crawl delay")
    
    args = parser.parse_args()
    
    if args.summary:
        checker = RobotsCompliantCrawler(args.user_agent)
        summary = checker.get_domain_summary(args.url)
        print("Robots.txt Summary:")
        print(json.dumps(summary, indent=2))
    
    elif args.delay:
        delay = get_crawl_delay(args.url, args.user_agent)
        print(f"Crawl delay for {args.url}: {delay}s")
    
    else:
        allowed = check_robots_compliance(args.url, args.user_agent)
        print(f"URL {args.url}: {'ALLOWED' if allowed else 'DISALLOWED'}")