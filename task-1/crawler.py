import undetected_chromedriver as uc
import time
import sqlite3
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import threading
import re
import os
import logging

# Import robots.txt compliance module
try:
    from robots_parser import RobotsCompliantCrawler
    ROBOTS_AVAILABLE = True
except ImportError:
    print("WARNING: robots_parser.py not found. Robots.txt compliance disabled.")
    ROBOTS_AVAILABLE = False

BASE_URL = "https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/"
DB_PATH = "publications.db"
JSON_PATH = "publications.json"

db_lock = threading.Lock()

# Initialize robots.txt compliance checker
if ROBOTS_AVAILABLE:
    robots_checker = RobotsCompliantCrawler(
        user_agent="Academic-Crawler/1.0 (Coventry University Research)",
        respect_robots=True
    )
else:
    robots_checker = None

def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT,
            authors TEXT,  -- JSON array
            abstract TEXT,
            year TEXT,
            UNIQUE(title, year)
        )''')
        conn.commit()
        conn.close()

def save_publication(title, link, authors_array, abstract, year):
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            authors_json = json.dumps(authors_array) if authors_array else "[]"
            c.execute(
                "INSERT OR IGNORE INTO publications (title, link, authors, abstract, year) VALUES (?, ?, ?, ?, ?)",
                (title, link, authors_json, abstract, year)
            )
            conn.commit()
            print(f"SAVED: {title[:30]}... | Authors: {len(authors_array)} | Year: {year} | Abstract: {len(abstract)} chars")
            return True
        except sqlite3.Error as e:
            print(f"ERROR: DB Error: {e}")
            return False
        finally:
            conn.close()

def create_driver(headless=True, worker_id=None):
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1366,768")
    
    # Additional options to help with Cloudflare detection
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Set realistic user agent
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    if worker_id is not None:
        user_data_dir = f"C:\\temp\\chrome_{worker_id}" if os.name == 'nt' else f"/tmp/chrome_{worker_id}"
        options.add_argument(f"--user-data-dir={user_data_dir}")
    
    try:
        driver = uc.Chrome(options=options, version_main=None)
        
        # Execute script to remove webdriver properties
        try:
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except:
            pass  # Ignore if this fails
        
        return driver
    except Exception as e:
        print(f"ERROR: Driver creation failed: {e}")
        return None

def handle_setup(driver):
    print("WAITING: Handling Cloudflare and cookies...")
    
    # Cloudflare bypass
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            if "coventry.ac.uk" in driver.current_url and "cloudflare" not in driver.page_source.lower():
                print("SUCCESS: Cloudflare bypassed")
                break
        except:
            pass
        time.sleep(2)
    
    # Cookie popup
    try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
        )
        cookie_btn.click()
        print("SUCCESS: Cookies handled")
        time.sleep(2)
    except:
        print("INFO: No cookie popup")
    
    return True

def check_robots_compliance(url):
    """Check if URL is allowed by robots.txt and handle crawl delays"""
    if not ROBOTS_AVAILABLE or not robots_checker:
        return True
    
    try:
        # Check if URL is allowed
        if not robots_checker.can_crawl(url):
            print(f"ROBOTS: URL blocked by robots.txt: {url}")
            return False
        
        # Wait if required by crawl delay
        robots_checker.pre_crawl_wait(url)
        print(f"ROBOTS: Compliance check passed for {url}")
        return True
        
    except Exception as e:
        print(f"ROBOTS: Error checking compliance for {url}: {e}")
        return True  # Allow by default if there's an error

def get_robots_summary():
    """Get robots.txt compliance summary"""
    if not ROBOTS_AVAILABLE or not robots_checker:
        return {"status": "Robots.txt compliance disabled"}
    
    try:
        stats = robots_checker.get_crawl_stats()
        domain_summary = robots_checker.get_domain_summary(BASE_URL)
        
        return {
            "status": "Robots.txt compliance enabled",
            "base_url": BASE_URL,
            "crawl_stats": stats,
            "domain_rules": domain_summary
        }
    except Exception as e:
        return {"status": f"Error getting robots summary: {e}"}

def extract_from_listing_page(driver):
    """STEP 1: Extract title, link, year from listing page only"""
    print("\nSTEP 1: Extracting basic info from LISTING PAGE...")
    time.sleep(3)
    
    publications = []
    
    try:
        containers = driver.find_elements(By.CSS_SELECTOR, ".result-container")
        print(f"Found {len(containers)} publication containers on listing page")
        
        if len(containers) == 0:
            print("ERROR: No containers found! Page might not be loaded properly.")
            # Debug: save page source
            with open("debug_listing_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("SAVED: Listing page source to debug_listing_page.html")
            return []
        
        for i in range(len(containers)):
            try:
                containers = driver.find_elements(By.CSS_SELECTOR, ".result-container")
                if i >= len(containers):
                    break
                container = containers[i]
                
                print(f"  Processing container {i+1}/{len(containers)}")
                
                # Extract title and link
                try:
                    title_elem = container.find_element(By.CSS_SELECTOR, "h3.title a.link span")
                    link_elem = container.find_element(By.CSS_SELECTOR, "h3.title a.link")
                    
                    title = title_elem.text.strip()
                    link = link_elem.get_attribute("href")
                    
                    if not title or not link:
                        print(f"    ERROR: Missing title or link")
                        continue
                    
                    # Make URL absolute
                    if not link.startswith("http"):
                        link = urljoin(BASE_URL, link)
                    
                    print(f"    Title: {title[:40]}...")
                    print(f"    Link: {link}")
                    
                    # Extract year from listing page
                    year = ""
                    try:
                        date_elem = container.find_element(By.CSS_SELECTOR, "span.date")
                        date_text = date_elem.text.strip()
                        year_matches = re.findall(r'\b((?:19|20)\d{2})\b', date_text)
                        if year_matches:
                            year = year_matches[0]
                        print(f"    Year: {year}")
                    except:
                        print(f"    WARNING: No year found")
                    
                    publications.append({
                        'title': title,
                        'link': link,
                        'year': year
                    })
                    
                except NoSuchElementException as e:
                    print(f"    ERROR: Could not find title/link elements: {e}")
                    continue
                
            except Exception as e:
                print(f"    ERROR: Container {i+1} error: {e}")
                continue
        
        print(f"\nSINGLE PAGE COMPLETE: Extracted {len(publications)} publications from current listing page")
        return publications
        
    except Exception as e:
        print(f"ERROR: Listing page extraction error: {e}")
        return []

def extract_from_listing_pages(driver, max_pages=None):
    """STEP 1: Extract title, link, year from all listing pages"""
    print("\nSTEP 1: Extracting basic info from ALL LISTING PAGES...")
    
    all_publications = []
    page_num = 1
    
    while True:
        print(f"\n--- Processing PAGE {page_num} ---")
        
        publications = extract_from_listing_page(driver)
        if not publications:
            print(f"No publications found on page {page_num}, stopping...")
            break
        
        all_publications.extend(publications)
        print(f"Page {page_num}: Found {len(publications)} publications (Total: {len(all_publications)})")
        
        # Check if we've reached max pages
        if max_pages and page_num >= max_pages:
            print(f"Reached maximum pages limit ({max_pages})")
            break
        
        # Try to find and click next page button
        if not go_to_next_page(driver, page_num):
            print("No more pages available")
            break
        
        page_num += 1
        time.sleep(3)  # Delay between pages
    
    print(f"\nSTEP 1 COMPLETE: Extracted {len(all_publications)} publications from {page_num} pages")
    print("NOTE: Authors and abstracts will be extracted from individual pages next")
    return all_publications

def go_to_next_page(driver, current_page):
    """Navigate to the next page of results"""
    try:
        next_page_num = current_page + 1
        print(f"Looking for page {next_page_num}...")
        
        # Look for pagination links with .step class
        page_links = driver.find_elements(By.CSS_SELECTOR, "a.step")
        
        for link in page_links:
            try:
                text = link.text.strip()
                href = link.get_attribute("href")
                
                # Check if this link is for the next page number
                if text == str(next_page_num):
                    print(f"Found page {next_page_num} link: {href}")
                    driver.execute_script("arguments[0].click();", link)
                    time.sleep(5)  # Wait for page to load
                    return True
                    
                # Also check URL for page parameter
                elif href and f"page={current_page}" in href:
                    print(f"Found next page via URL pattern: {href}")
                    driver.execute_script("arguments[0].click();", link)
                    time.sleep(5)
                    return True
                    
            except Exception as e:
                print(f"Error checking pagination link: {e}")
                continue
        
        # Alternative approach: try to construct the next page URL directly
        try:
            current_url = driver.current_url
            if "page=" in current_url:
                # Replace the page parameter
                import re
                next_url = re.sub(r'page=\d+', f'page={current_page}', current_url)
            else:
                # Add page parameter
                separator = '&' if '?' in current_url else '?'
                next_url = f"{current_url}{separator}page={current_page}"
            
            print(f"Trying direct URL navigation: {next_url}")
            driver.get(next_url)
            time.sleep(5)
            
            # Verify we got to the next page by checking if publications exist
            containers = driver.find_elements(By.CSS_SELECTOR, ".result-container")
            if len(containers) > 0:
                print(f"Successfully navigated to page {next_page_num} via direct URL")
                return True
            else:
                print(f"No publications found on page {next_page_num} - might be last page")
                return False
                
        except Exception as e:
            print(f"Direct URL navigation failed: {e}")
        
        print(f"Could not find way to navigate to page {next_page_num}")
        return False
        
    except Exception as e:
        print(f"ERROR: Failed to go to next page: {e}")
        return False

def extract_from_individual_page(pub_info, worker_id):
    """STEP 2: Navigate to individual publication page to get authors and abstract"""
    individual_driver = None
    try:
        title = pub_info['title']
        link = pub_info['link']
        year = pub_info['year']
        
        print(f"\nWorker {worker_id}: STEP 2 - Navigating to INDIVIDUAL page")
        print(f"    Title: {title[:40]}...")
        print(f"    URL: {link}")
        
        # Check robots.txt compliance before crawling
        if not check_robots_compliance(link):
            print(f"    SKIPPED: URL blocked by robots.txt")
            return {'success': False, 'reason': 'blocked_by_robots', 'title': title, 'link': link}
        
        # Create separate driver for this individual page
        individual_driver = create_driver(headless=True, worker_id=worker_id)
        if not individual_driver:
            print(f"    ERROR: Failed to create driver for individual page")
            return None
        
        print(f"    Navigating to individual publication page...")
        individual_driver.get(link)
        
        # Handle Cloudflare challenge on individual page
        print(f"    Checking for Cloudflare challenge...")
        cloudflare_bypassed = False
        start_time = time.time()
        
        while time.time() - start_time < 45:  # Extended timeout for Cloudflare
            try:
                page_source = individual_driver.page_source.lower()
                current_url = individual_driver.current_url
                
                # Check if we're on Cloudflare challenge page
                if "cloudflare" in page_source or "just a moment" in page_source or "verify you are human" in page_source:
                    print(f"    WAITING: Cloudflare challenge detected, waiting...")
                    time.sleep(3)
                    continue
                
                # Check if we've reached the actual publication page
                if "coventry.ac.uk" in current_url and "publications" in current_url:
                    print(f"    SUCCESS: Cloudflare challenge bypassed successfully")
                    cloudflare_bypassed = True
                    break
                    
            except Exception as e:
                print(f"    WARNING: Error checking Cloudflare status: {e}")
                time.sleep(2)
                continue
        
        if not cloudflare_bypassed:
            print(f"    ERROR: Failed to bypass Cloudflare challenge after 45 seconds")
            # Save debug page
            with open(f"debug_individual_page_worker_{worker_id}.html", "w", encoding="utf-8") as f:
                f.write(individual_driver.page_source)
            return None
        
        # Handle cookie popup on individual page
        try:
            cookie_btn = WebDriverWait(individual_driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
            )
            cookie_btn.click()
            time.sleep(1)
            print(f"    SUCCESS: Handled cookie popup on individual page")
        except:
            print(f"    INFO: No cookie popup on individual page")
        
        # Wait for individual page content to load
        try:
            WebDriverWait(individual_driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            print(f"    SUCCESS: Individual page content loaded successfully")
        except TimeoutException:
            print(f"    ERROR: Timeout loading individual page content")
            return None
        
        # Verify we're on the right page
        current_url = individual_driver.current_url
        page_title = individual_driver.title
        print(f"    Current URL: {current_url}")
        print(f"    Page title: {page_title[:50]}...")
        
        # EXTRACT AUTHORS FROM INDIVIDUAL PAGE
        print(f"    Extracting authors from individual page...")
        authors_array = []
        
        try:
            # Look for the specific authors element
            authors_elem = individual_driver.find_element(By.CSS_SELECTOR, "p.relations.persons")
            authors_text = authors_elem.text.strip()
            print(f"    SUCCESS: Found authors element: <p class='relations persons'>")
            print(f"    Raw authors text: '{authors_text}'")
            
            if authors_text:
                # Parse authors - handle both comma and & separators
                if '&' in authors_text:
                    # Split by & first
                    parts = authors_text.split('&')
                    for part in parts[:-1]:
                        sub_authors = [a.strip() for a in part.split(',') if a.strip()]
                        authors_array.extend(sub_authors)
                    # Add last author after &
                    last_author = parts[-1].strip()
                    if last_author:
                        authors_array.append(last_author)
                else:
                    # Split by comma only
                    authors_array = [a.strip() for a in authors_text.split(',') if a.strip()]
                
                # Clean up author names
                authors_array = [author.strip().rstrip(',').strip() 
                               for author in authors_array if author.strip()]
                
                print(f"    SUCCESS: Parsed {len(authors_array)} authors: {authors_array}")
            
        except NoSuchElementException:
            print(f"    ERROR: No authors found with selector: p.relations.persons")
            # Save page source for debugging
            with open(f"debug_individual_page_worker_{worker_id}.html", "w", encoding="utf-8") as f:
                f.write(individual_driver.page_source)
            print(f"    SAVED: Individual page source for debugging")
        
        except Exception as e:
            print(f"    ERROR: Authors extraction error: {e}")
        
        # EXTRACT ABSTRACT FROM INDIVIDUAL PAGE
        print(f"    Extracting abstract from individual page...")
        abstract = ""
        
        try:
            abstract_elem = individual_driver.find_element(By.CSS_SELECTOR, "div.textblock")
            abstract = abstract_elem.text.strip()
            print(f"    SUCCESS: Found abstract element: <div class='textblock'>")
            print(f"    Abstract length: {len(abstract)} characters")
            
            if len(abstract) < 50:
                print(f"    WARNING: Abstract seems too short, might not be correct")
            
        except NoSuchElementException:
            print(f"    ERROR: No abstract found with selector: div.textblock")
            
        except Exception as e:
            print(f"    ERROR: Abstract extraction error: {e}")
        
        # SAVE TO DATABASE
        print(f"    Saving to database...")
        saved = save_publication(title, link, authors_array, abstract, year)
        
        if saved:
            print(f"    SUCCESS: Successfully saved publication with {len(authors_array)} authors")
        
        return {
            'title': title,
            'link': link,
            'authors': authors_array,
            'abstract': abstract,
            'year': year,
            'success': True
        }
        
    except Exception as e:
        print(f"    ERROR: Individual page processing error: {e}")
        return None
    
    finally:
        if individual_driver:
            try:
                individual_driver.quit()
                print(f"    Closed individual page driver for worker {worker_id}")
            except:
                pass

def process_all_individual_pages(publications, max_workers=2):
    """Process all individual publication pages in parallel"""
    if not publications:
        return []
    
    print(f"\nSTEP 2: Processing {len(publications)} INDIVIDUAL pages with {max_workers} workers...")
    print("NOTE: Each worker will navigate to individual publication pages to extract authors & abstracts")
    print("NOTE: Reduced workers and added delays to avoid Cloudflare rate limiting")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit individual page processing tasks with delays
        future_to_pub = {}
        for i, pub in enumerate(publications):
            # Add small delay between submissions to avoid overwhelming Cloudflare
            time.sleep(2)  # 2 second delay between starting workers
            future = executor.submit(extract_from_individual_page, pub, i % max_workers)
            future_to_pub[future] = pub
        
        completed = 0
        for future in as_completed(future_to_pub):
            completed += 1
            result = future.result()
            if result:
                results.append(result)
            
            # Progress tracking
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = len(publications) - completed
            eta = remaining / rate if rate > 0 else 0
            
            print(f"Progress: {completed}/{len(publications)} individual pages "
                  f"({completed/len(publications)*100:.1f}%) Rate: {rate:.1f}/s ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    successful = len(results)
    
    print(f"\nSTEP 2 COMPLETE: {successful}/{len(publications)} individual pages processed in {elapsed:.1f}s")
    print(f"Average: {elapsed/len(publications):.1f}s per individual page")
    
    return results

def main_corrected_crawl(headless=False, max_pages=1, max_workers=2):
    """Main crawling function with clear individual page processing"""
    print("CORRECTED INDIVIDUAL PAGE SCRAPER")
    print("="*60)
    print("PROCESS:")
    print("  STEP 1: Extract title, link, year from LISTING page")
    print("  STEP 2: Navigate to each INDIVIDUAL publication page")
    print("         - Extract authors from: <p class='relations persons'>")
    print("         - Extract abstract from: <div class='textblock'>")
    print("="*60)
    
    init_db()
    
    # Display robots.txt compliance status
    robots_summary = get_robots_summary()
    print(f"\\nROBOTS.TXT COMPLIANCE STATUS:")
    print(f"  Status: {robots_summary.get('status', 'Unknown')}")
    if 'domain_rules' in robots_summary and robots_summary['domain_rules']:
        domain_rules = robots_summary['domain_rules']
        print(f"  Domain: {domain_rules.get('domain', 'Unknown')}")
        print(f"  Crawl Delay: {domain_rules.get('crawl_delay', 'None')}s")
        print(f"  Sitemaps Found: {len(domain_rules.get('sitemaps', []))}")
    print("="*60)
    # Check robots.txt compliance for base URL
    if not check_robots_compliance(BASE_URL):
        print(f"ERROR: Base URL {BASE_URL} is blocked by robots.txt")
        print("Crawling cannot proceed due to robots.txt restrictions")
        return
    
    main_driver = create_driver(headless)
    
    if not main_driver:
        return
    
    try:
        print(f"Navigating to LISTING page: {BASE_URL}")
        main_driver.get(BASE_URL)
        
        if not handle_setup(main_driver):
            return
        
        # STEP 1: Extract from all listing pages
        publications = extract_from_listing_pages(main_driver, max_pages)
        
        if not publications:
            print("ERROR: No publications found on listing pages")
            return
        
        print(f"\nSTEP 1 RESULT: Found {len(publications)} publications from all pages")
        print("Moving to STEP 2: Processing individual pages...")
        
        # STEP 2: Process individual pages in parallel
        results = process_all_individual_pages(publications, max_workers)
        
        print(f"\nCRAWLING COMPLETED!")
        print(f"Publications from listing: {len(publications)}")
        print(f"Individual pages processed: {len(results)}")
        
        # Export to JSON
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT title, authors, abstract, year, link FROM publications ORDER BY ROWID DESC")
            rows = c.fetchall()
            conn.close()
        
        publications_data = []
        for row in rows:
            try:
                authors_array = json.loads(row[1]) if row[1] else []
            except:
                authors_array = []
            
            publications_data.append({
                "title": row[0] or "",
                "authors": authors_array,
                "abstract": row[2] or "",
                "year": row[3] or "",
                "link": row[4] or ""
            })
        
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(publications_data, f, ensure_ascii=False, indent=2)
        
        print(f"SUCCESS: Exported {len(publications_data)} publications to {JSON_PATH}")
        
        # Show successful extractions
        successful_with_authors = [pub for pub in publications_data if pub['authors']]
        print(f"Publications with authors extracted: {len(successful_with_authors)}")
        
        if successful_with_authors:
            print("\nSample publications with authors:")
            for i, pub in enumerate(successful_with_authors[:3], 1):
                print(f"  {i}. {pub['title'][:35]}... ({pub['year']})")
                print(f"     Authors: {pub['authors']}")
        
    except Exception as e:
        print(f"ERROR: Crawling error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Closing main browser...")
        try:
            main_driver.quit()
        except:
            pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Corrected Individual Page Scraper")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl (default: all pages)")
    parser.add_argument("--max-workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    headless_mode = args.headless and not args.debug
    
    main_corrected_crawl(
        headless=headless_mode,
        max_pages=args.max_pages,
        max_workers=args.max_workers
    )