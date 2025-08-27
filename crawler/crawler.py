#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, time, re
from math import ceil
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse, urljoin
import requests
from urllib import robotparser

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

PURE_PORTAL_BASE_URL = (
    "https://pureportal.coventry.ac.uk/en/organisations/"
    "fbl-school-of-economics-finance-and-accounting/publications/"
)

# =========================== Browser Configuration Helpers ===========================
def configure_chrome_options(is_headless: bool, use_legacy_headless: bool = False) -> Options:
    chrome_options = Options()
    if is_headless:
        chrome_options.add_argument("--headless" + ("" if use_legacy_headless else "=new"))
    chrome_options.add_argument("--window-size=1366,900")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--no-default-browser-check")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-features=CalculateNativeWinOcclusion,MojoVideoDecoder")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.page_load_strategy = "eager"
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
    return chrome_options

def initialize_chrome_driver(is_headless: bool, use_legacy_headless: bool = False) -> webdriver.Chrome:
    service = ChromeService(ChromeDriverManager().install(), log_output=os.devnull)
    browser = webdriver.Chrome(service=service, options=configure_chrome_options(is_headless, use_legacy_headless))
    browser.set_page_load_timeout(45)
    try:
        browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception:
        pass
    return browser

def handle_cookie_consent(browser: webdriver.Chrome):
    try:
        consent_button = WebDriverWait(browser, 6).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        browser.execute_script("arguments[0].click();", consent_button)
        time.sleep(0.25)
    except (TimeoutException, Exception):
        pass

# =========================== Publication Listing (Stage 1) ===========================
def extract_publications_from_page(browser: webdriver.Chrome, page_number: int) -> List[Dict]:
    page_url = f"{PURE_PORTAL_BASE_URL}?page={page_number}"
    browser.get(page_url)
    try:
        WebDriverWait(browser, 15).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, ".result-container h3.title a")
                      or "No results" in d.page_source
        )
    except TimeoutException:
        pass

    publication_cards = browser.find_elements(By.CLASS_NAME, "result-container")
    publication_data: List[Dict] = []
    for card in publication_cards:
        try:
            link_element = card.find_element(By.CSS_SELECTOR, "h3.title a")
            publication_title = link_element.text.strip()
            publication_url = link_element.get_attribute("href")
            if publication_title and publication_url:
                publication_data.append({"title": publication_title, "link": publication_url})
        except Exception:
            continue
    return publication_data

def gather_all_listing_links(max_pages: int, headless_listing: bool = False, legacy_headless: bool = False) -> List[Dict]:
    # Listing works more reliably non-headless
    driver = initialize_chrome_driver(is_headless=headless_listing, use_legacy_headless=legacy_headless)
    try:
        driver.get(PURE_PORTAL_BASE_URL)
        handle_cookie_consent(driver)
        all_rows: List[Dict] = []
        for i in range(max_pages):
            print(f"[LIST] Page {i+1}/{max_pages}")
            rows = extract_publications_from_page(driver, i)
            if not rows:
                print(f"[LIST] Empty at page index {i}; stopping early.")
                break
            all_rows.extend(rows)
        # dedupe by link
        uniq = {}
        for r in all_rows:
            uniq[r["link"]] = r
        return list(uniq.values())
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# =========================== Detail Extraction (Stage 2) ===========================
# =========================== Detail Page Extraction Helpers ===========================
DIGIT_PATTERN = re.compile(r"\d")
AUTHOR_NAME_PATTERN = re.compile(
    r"[A-Z][A-Za-z''\-]+,\s*(?:[A-Z](?:\.)?)(?:\s*[A-Z](?:\.)?)*",
    flags=re.UNICODE
)

def remove_duplicates(sequence: List[str]) -> List[str]:
    """Remove duplicate entries while preserving order"""
    unique_items, result = set(), []
    for item in sequence:
        item = item.strip()
        if item and item not in unique_items:
            unique_items.add(item)
            result.append(item)
    return result

def extract_metadata_values(driver: webdriver.Chrome, meta_attributes: List[str]) -> List[str]:
    """Extract values from meta tags with given names/properties"""
    values = []
    for attr in meta_attributes:
        for element in driver.find_elements(By.CSS_SELECTOR, f'meta[name="{attr}"], meta[property="{attr}"]'):
            content = (element.get_attribute("content") or "").strip()
            if content:
                values.append(content)
    return remove_duplicates(values)

def extract_authors_from_json_ld(driver: webdriver.Chrome) -> List[str]:
    """Extract author names from JSON-LD structured data"""
    import json as _json
    author_names = []
    for script in driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]'):
        content = (script.get_attribute("textContent") or "").strip()
        if not content:
            continue
        try:
            data = _json.loads(content)
            json_objects = data if isinstance(data, list) else [data]
            for obj in json_objects:
                authors = obj.get("author")
                if not authors:
                    continue
                if isinstance(authors, list):
                    for author in authors:
                        name = author.get("name") if isinstance(author, dict) else str(author)
                        if name: 
                            author_names.append(name)
                elif isinstance(authors, dict):
                    name = authors.get("name")
                    if name: 
                        author_names.append(name)
                elif isinstance(authors, str):
                    author_names.append(authors)
        except Exception:
            continue
    return remove_duplicates(author_names)

def expand_author_list(driver: webdriver.Chrome):
    """Expand collapsed author lists if present"""
    try:
        expand_buttons = driver.find_elements(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show') or "
            "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'more')]"
        )
        for button in expand_buttons[:2]:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
                time.sleep(0.15)
                button.click()
                time.sleep(0.25)
            except Exception:
                continue
    except Exception:
        pass

def extract_detail_for_link(driver: webdriver.Chrome, link: str, title_hint: str, delay: float) -> Dict:
    driver.get(link)
    handle_cookie_consent(driver)
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    except TimeoutException:
        pass

    # Title (use detail title if available; else listing hint)
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except NoSuchElementException:
        title = title_hint or ""

    # Try to reveal hidden lists
    expand_author_list(driver)

    # AUTHORS: DOM → subtitle simple → meta → JSON-LD
    authors = []
    for sel in [
        ".relations.persons a[href*='/en/persons/'] span",
        ".relations.persons a[href*='/en/persons/']",
        "section#persons a[href*='/en/persons/'] span",
        "section#persons a[href*='/en/persons/']",
    ]:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            t = el.text.strip()
            if t:
                authors.append(t)
        if authors:
            break
    if not authors:
        authors = extract_authors_from_subtitle(driver, title)
    if not authors:
        authors = extract_metadata_values(driver, ["citation_author", "dc.contributor", "dc.contributor.author"])
    if not authors:
        authors = extract_authors_from_json_ld(driver)
    authors = remove_duplicates(authors)

    # PUBLISHED DATE
    published_date = None
    for sel in ["span.date", "time[datetime]", "time"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            published_date = el.get_attribute("datetime") or el.text.strip()
            if published_date:
                break
        except NoSuchElementException:
            continue
    if not published_date:
        metas = extract_metadata_values(driver, ["citation_publication_date", "dc.date", "article:published_time"])
        if metas:
            published_date = metas[0]

    # ABSTRACT
    abstract_txt = None
    for sel in [
        "section#abstract .textblock",
        "section.abstract .textblock",
        "div.abstract .textblock",
        "div#abstract",
        "section#abstract",
        "div.textblock",
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            txt = el.text.strip()
            if txt and len(txt) > 15:
                abstract_txt = txt
                break
        except NoSuchElementException:
            continue
    if not abstract_txt:
        try:
            hdrs = driver.find_elements(By.CSS_SELECTOR, "h2, h3")
            for h in hdrs:
                if "abstract" in h.text.strip().lower():
                    nxt = h.find_element(By.XPATH, "./following::*[self::div or self::p or self::section][1]")
                    txt = nxt.text.strip()
                    if txt:
                        abstract_txt = txt
                        break
        except Exception:
            pass

    time.sleep(delay)  # polite delay
    return {
        "title": title,
        "link": link,
        "authors": authors,
        "published_date": published_date,
        "abstract": abstract_txt or ""
    }

def extract_publication_details(driver: webdriver.Chrome, publication_url: str, title_hint: str, delay: float) -> Dict:
    """Extract detailed information for a single publication"""
    driver.get(publication_url)
    handle_cookie_consent(driver)
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    except TimeoutException:
        pass

    # Extract main title
    try:
        main_title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except NoSuchElementException:
        main_title = title_hint or ""

    # Expand any collapsed author sections
    expand_author_list(driver)

    # Extract authors using multiple strategies
    author_list = []
    author_selectors = [
        ".relations.persons a[href*='/en/persons/'] span",
        ".relations.persons a[href*='/en/persons/']",
        "section#persons a[href*='/en/persons/'] span",
        "section#persons a[href*='/en/persons/']",
    ]
    
    for selector in author_selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            author_name = element.text.strip()
            if author_name:
                author_list.append(author_name)
        if author_list:
            break
            
    if not author_list:
        author_list = extract_authors_from_subtitle(driver, main_title)
    if not author_list:
        author_list = extract_metadata_values(driver, [
            "citation_author", "dc.contributor", "dc.contributor.author"
        ])
    if not author_list:
        author_list = extract_authors_from_json_ld(driver)
    author_list = remove_duplicates(author_list)

    # Extract publication date
    publication_date = extract_publication_date(driver)

    # Extract abstract content
    abstract_content = extract_abstract_content(driver)

    time.sleep(delay)  # Polite delay between requests
    
    return {
        "title": main_title,
        "link": publication_url,
        "authors": author_list,
        "published_date": publication_date,
        "abstract": abstract_content or ""
    }

def extract_publication_date(driver: webdriver.Chrome) -> str:
    """Extract publication date using multiple strategies"""
    date_selectors = ["span.date", "time[datetime]", "time"]
    
    for selector in date_selectors:
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, selector)
            date_value = date_element.get_attribute("datetime") or date_element.text.strip()
            if date_value:
                return date_value
        except NoSuchElementException:
            continue
            
    date_metadata = extract_metadata_values(driver, [
        "citation_publication_date", "dc.date", "article:published_time"
    ])
    return date_metadata[0] if date_metadata else None

def extract_abstract_content(driver: webdriver.Chrome) -> str:
    """Extract abstract content using multiple strategies"""
    abstract_selectors = [
        "section#abstract .textblock",
        "section.abstract .textblock",
        "div.abstract .textblock",
        "div#abstract",
        "section#abstract",
        "div.textblock",
    ]
    
    for selector in abstract_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            content = element.text.strip()
            if content and len(content) > 15:
                return content
        except NoSuchElementException:
            continue
            
    try:
        headers = driver.find_elements(By.CSS_SELECTOR, "h2, h3")
        for header in headers:
            if "abstract" in header.text.strip().lower():
                next_element = header.find_element(
                    By.XPATH, 
                    "./following::*[self::div or self::p or self::section][1]"
                )
                content = next_element.text.strip()
                if content:
                    return content
    except Exception:
        pass
    
    return ""

def extract_authors_from_subtitle(driver: webdriver.Chrome, title: str) -> List[str]:
    """Extract author names from the subtitle section using pattern matching"""
    try:
        subtitle_element = driver.find_element(By.CSS_SELECTOR, "div.subtitle")
        subtitle_text = subtitle_element.text.strip()
        
        # Remove the title from subtitle if present to avoid false matches
        if title:
            subtitle_text = subtitle_text.replace(title, "")
            
        # Find all matches of author name pattern
        author_matches = AUTHOR_NAME_PATTERN.finditer(subtitle_text)
        authors = [match.group().strip() for match in author_matches]
        return remove_duplicates(authors)
    except NoSuchElementException:
        return []

# =========================== Worker Functions ===========================
def process_detail_batch(batch: List[Dict], headless: bool, legacy_headless: bool, delay: float) -> List[Dict]:
    driver = initialize_chrome_driver(is_headless=headless, use_legacy_headless=legacy_headless)
    extracted_details: List[Dict] = []
    robot_parser = RobotParser()
    
    try:
        for index, publication in enumerate(batch, 1):
            if not robot_parser.can_fetch(publication["link"]):
                print(f"[WORKER] Skipping {publication['link']} (blocked by robots.txt)")
                continue
            try:
                publication_details = extract_publication_details(
                    driver, 
                    publication["link"], 
                    publication.get("title", ""), 
                    delay
                )
                extracted_details.append(publication_details)
                print(f"[WORKER] {index}/{len(batch)} OK: {publication_details['title'][:60]}")
            except WebDriverException as e:
                print(f"[WORKER] ERR {publication['link']}: {e}")
                continue
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    return extracted_details

def split_into_batches(items: List[Dict], batch_count: int) -> List[List[Dict]]:
    """Split items into approximately equal-sized batches"""
    if batch_count <= 1:
        return [items]
    batch_size = ceil(len(items) / batch_count)
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

# =========================== Robots.txt Handling ===========================
class RobotParser:
    def __init__(self):
        self.parsers = {}
        # More generic user agent
        self.user_agent = "*"  # Use wildcard agent since that's typically allowed

    def load_robots_txt(self, url: str) -> None:
        domain = urlparse(url).scheme + "://" + urlparse(url).netloc
        if domain not in self.parsers:
            parser = robotparser.RobotFileParser()
            parser.set_url(urljoin(domain, "/robots.txt"))
            try:
                # Print robots.txt content for debugging
                robots_url = urljoin(domain, "/robots.txt")
                response = requests.get(robots_url, timeout=10)
                print(f"[ROBOTS] Content from {robots_url}:")
                print(response.text)
                parser.parse(response.text.splitlines())
                self.parsers[domain] = parser
            except Exception as e:
                print(f"[ROBOTS] Warning: Could not fetch robots.txt from {domain}: {e}")
                empty_parser = robotparser.RobotFileParser()
                empty_parser.parse(["User-agent: *", "Allow: /"])
                self.parsers[domain] = empty_parser

    def can_fetch(self, url: str, ignore_robots: bool = False) -> bool:
        if ignore_robots:
            return True
        self.load_robots_txt(url)
        domain = urlparse(url).scheme + "://" + urlparse(url).netloc
        return self.parsers[domain].can_fetch(self.user_agent, url)

# =========================== Main Orchestrator ===========================
def crawl():
    """Main orchestrator function for the crawler"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Coventry PurePortal publication crawler (extracts listings, authors, abstracts, and dates)."
    )
    parser.add_argument("--outdir", default="data", help="Directory to store output files")
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=50, 
        help="Maximum listing pages to scan (stops early if empty page found)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=8, 
        help="Number of parallel headless browsers for detail extraction"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.35, 
        help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--listing-headless", 
        action="store_true", 
        help="Run listing phase in headless mode (not recommended)"
    )
    parser.add_argument(
        "--legacy-headless", 
        action="store_true", 
        help="Use legacy headless mode instead of new headless"
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt restrictions (use responsibly)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create robot parser
    robot_parser = RobotParser()
    
    # Check if we can crawl the base URL
    if not robot_parser.can_fetch(PURE_PORTAL_BASE_URL, args.ignore_robots):
        print("Crawling not allowed by robots.txt. Use --ignore-robots to override (use responsibly)")
        return

    # Stage 1: Collect publication listings
    print(f"[STAGE 1] Collecting publication links (max {args.max_pages} pages)...")
    publication_links = gather_all_listing_links(
        args.max_pages, 
        headless_listing=args.listing_headless, 
        legacy_headless=args.legacy_headless
    )

    # Filter out URLs that are not allowed by robots.txt
    publication_links = [
        link for link in publication_links 
        if robot_parser.can_fetch(link["link"], args.ignore_robots)
    ]
    
    # Save listing results
    listing_file = output_dir / "publications_links.json"
    listing_file.write_text(json.dumps(publication_links, indent=2), encoding="utf-8")
    print(f"[STAGE 1] Collected {len(publication_links)} unique publication links.")

    # Stage 2: Extract detailed information
    print(f"[STAGE 2] Extracting publication details using {args.workers} parallel workers...")
    publication_batches = split_into_batches(publication_links, args.workers)
    detailed_results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_batch = {
            executor.submit(
                process_detail_batch, 
                batch, 
                True, 
                args.legacy_headless, 
                args.delay
            ): batch for batch in publication_batches
        }
        
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_results = future.result() or []
            detailed_results.extend(batch_results)
            completed_batches += 1
            print(f"[STAGE 2] Completed {completed_batches}/{len(publication_batches)} "
                  f"batches (+{len(batch_results)} items)")

    # Consolidate and save results
    publications_by_link: Dict[str, Dict] = {}
    
    # First pass: basic information from listings
    for item in publication_links:
        publications_by_link[item["link"]] = {
            "title": item["title"], 
            "link": item["link"]
        }
    
    # Second pass: override with detailed information where available
    for record in detailed_results:
        publications_by_link[record["link"]] = record

    final_publications = list(publications_by_link.values())
    output_file = output_dir / "publications.json"
    output_file.write_text(
        json.dumps(final_publications, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    print(f"[DONE] Successfully saved {len(final_publications)} publication records to {output_file}")

if __name__ == "__main__":
    crawl()
