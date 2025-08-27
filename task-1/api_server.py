from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import json
import time
import asyncio
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import functools
from datetime import datetime
from contextlib import asynccontextmanager
import logging

from search_engine import SearchEngine, SearchResult, Publication
from query_processor import QueryParser, QueryValidator, QueryExpander
from ranking_system import RankingSystem, PersonalizedRanking, DiversityRanking
from crawler_scheduler import CrawlerScheduler, CrawlFrequency
from crawler import get_robots_summary

# Pydantic models for API
class PublicationModel(BaseModel):
    id: int
    title: str
    authors: List[str]
    abstract: str
    year: str
    link: str

class SearchResultModel(BaseModel):
    publication: PublicationModel
    score: float
    matched_fields: List[str]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: Optional[Dict[str, Union[str, int, List[str]]]] = None
    sort_by: str = Field(default="relevance")
    sort_order: str = Field(default="desc")
    expand_query: bool = Field(default=False)
    diversify: bool = Field(default=True)
    user_id: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResultModel]
    total_results: int
    query_time: float
    query_info: Dict
    suggestions: List[str] = []
    facets: Dict[str, List[Dict]] = {}

class SuggestionsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=10)

class StatsResponse(BaseModel):
    total_publications: int
    total_terms: int
    index_size: int
    last_updated: Optional[str] = None

class CrawlJobRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    frequency: str = Field(..., description="hourly, daily, weekly, monthly, custom")
    max_pages: Optional[int] = Field(None, ge=1)
    max_workers: int = Field(default=2, ge=1, le=5)
    custom_schedule: Optional[str] = Field(None, description="Custom schedule string")
    enabled: bool = Field(default=True)

class CrawlJobResponse(BaseModel):
    id: str
    name: str
    frequency: str
    enabled: bool
    last_run: Optional[str]
    next_run: Optional[str]
    status: str
    max_pages: Optional[int]
    max_workers: int
    custom_schedule: Optional[str]
    created_at: str
    updated_at: str
    run_count: int
    success_count: int
    failure_count: int
    is_running: bool
    success_rate: float

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
search_engine = None
query_parser = None
query_validator = None
query_expander = None
ranking_system = None
personalized_ranking = None
diversity_ranking = None
crawler_scheduler = None

# Circuit breaker for search failures
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        logger.info("Circuit breaker reset")

circuit_breaker = CircuitBreaker()

# Thread pool for search execution - using context manager for better cleanup
search_executor = None

def get_search_executor():
    """Get or create search executor"""
    global search_executor
    if search_executor is None or search_executor._shutdown:
        search_executor = ThreadPoolExecutor(
            max_workers=2,  # Reduced to prevent resource exhaustion
            thread_name_prefix="search"
        )
    return search_executor

def execute_search_with_timeout(search_func, *args, timeout_seconds=8, **kwargs):
    """Execute search with timeout protection and cleanup"""
    executor = get_search_executor()
    future = None
    try:
        future = executor.submit(search_func, *args, **kwargs)
        result = future.result(timeout=timeout_seconds)
        return result if result is not None else []
    except FutureTimeoutError:
        logger.warning(f"Search timeout after {timeout_seconds}s - cancelling future")
        # Cancel the future to prevent resource leaks
        if future:
            future.cancel()
        return []
    except Exception as e:
        logger.error(f"Search execution error: {e}")
        # Cancel the future to prevent resource leaks  
        if future:
            future.cancel()
        return []
    finally:
        # Ensure we don't leave hanging futures
        if future and not future.done():
            try:
                future.cancel()
            except:
                pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global search_engine, query_parser, query_validator, query_expander
    global ranking_system, personalized_ranking, diversity_ranking, crawler_scheduler
    
    try:
        logger.info("Initializing Academic Search Engine API...")
        
        # Initialize components with error handling
        search_engine = SearchEngine()
        query_parser = QueryParser()
        query_validator = QueryValidator()
        query_expander = QueryExpander()
        ranking_system = RankingSystem()
        personalized_ranking = PersonalizedRanking()
        diversity_ranking = DiversityRanking()
        
        # Initialize crawler scheduler
        crawler_scheduler = CrawlerScheduler()
        crawler_scheduler.start_scheduler()
        
        logger.info(f"Search engine loaded with {search_engine.index.total_documents} publications")
        logger.info("Crawler scheduler started")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        # Continue with limited functionality
        yield
    finally:
        # Cleanup
        if crawler_scheduler:
            crawler_scheduler.stop_scheduler()
        
        # Shutdown search executor
        if search_executor:
            search_executor.shutdown(wait=True)
            
        logger.info("Shutting down Academic Search Engine API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Academic Search Engine API",
    description="REST API for academic publication search engine",
    version="1.0.0",
    lifespan=lifespan
)

# Add timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set timeout for requests - longer for search endpoints
        timeout = 45.0 if "/search" in str(request.url) else 30.0
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        logger.warning(f"Request timeout for {request.url}")
        return JSONResponse(
            status_code=408,
            content={
                "detail": "Request timeout",
                "message": "The request took too long to process. Please try again with a simpler query."
            }
        )
    except Exception as e:
        logger.error(f"Request error for {request.url}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "message": "An unexpected error occurred. Please try again."
            }
        )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get search engine with fallback
async def get_search_engine():
    global search_engine
    if search_engine is None:
        try:
            search_engine = SearchEngine()
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise HTTPException(status_code=503, detail="Search service unavailable")
    return search_engine

# Health check function
def check_search_health():
    """Check if search engine is healthy with timeout protection"""
    try:
        if not search_engine:
            return False
        if not hasattr(search_engine, 'index') or not search_engine.index:
            return False
        
        # Check if index has publications
        if not hasattr(search_engine.index, 'publications') or not search_engine.index.publications:
            logger.warning("Search index has no publications loaded")
            return False
        
        # Try a quick search with timeout
        results = execute_search_with_timeout(
            search_engine.search,
            "test",
            limit=1,
            timeout_seconds=3
        )
        return True
    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return False

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "message": "Academic Search Engine API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.post("/search", response_model=SearchResponse)
async def search_publications(
    request: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine)
):
    """Search for academic publications"""
    start_time = time.time()
    
    try:
        # Check service health
        if not check_search_health():
            raise HTTPException(status_code=503, detail="Search service not ready")
        
        # Initialize global components if needed
        global query_parser, query_validator, query_expander
        global personalized_ranking, diversity_ranking
        
        if not query_validator:
            query_validator = QueryValidator()
        if not query_parser:
            query_parser = QueryParser()
        if not query_expander:
            query_expander = QueryExpander()
        if not personalized_ranking:
            personalized_ranking = PersonalizedRanking()
        if not diversity_ranking:
            diversity_ranking = DiversityRanking()
        
        # Use circuit breaker for search operations
        def perform_search():
            # Check if search engine is properly initialized
            if not engine or not hasattr(engine, 'index') or not engine.index:
                logger.error("Search engine not properly initialized")
                raise HTTPException(status_code=503, detail="Search service unavailable")
            
            # Validate query
            validation = query_validator.validate(request.query)
            if not validation['valid']:
                raise HTTPException(
                    status_code=400, 
                    detail={"errors": validation['errors']}
                )
            
            # Parse query
            try:
                parsed_query = query_parser.parse(
                    validation['sanitized_query'], 
                    filters=request.filters
                )
            except Exception as e:
                logger.error(f"Query parsing error: {e}")
                raise HTTPException(status_code=400, detail="Invalid query format")
            
            # Expand query if requested
            if request.expand_query:
                try:
                    parsed_query = query_expander.expand_query(parsed_query)
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
                    # Continue without expansion
            
            # Execute search with timeout protection
            results = execute_search_with_timeout(
                engine.search,
                validation['sanitized_query'],
                limit=min(request.limit + request.offset, 1000),
                filters=request.filters,
                timeout_seconds=10
            )
            
            # Ensure results is always a list
            if results is None:
                results = []
            
            return validation, parsed_query, results
        
        # Execute search through circuit breaker
        try:
            validation, parsed_query, results = circuit_breaker.call(perform_search)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Circuit breaker caught error: {e}")
            # Return empty results with error info
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                query_info={
                    "original_query": request.query,
                    "error": "Search temporarily unavailable"
                },
                suggestions=[],
                facets={}
            )
        
        # Early return for empty results to avoid unnecessary processing
        if not results:
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                query_info={
                    "original_query": request.query,
                    "sanitized_query": validation['sanitized_query'] if validation else request.query,
                    "type": parsed_query.query_type if parsed_query else "simple"
                },
                suggestions=[],
                facets={}
            )
        
        # Apply post-processing with error handling
        try:
            # Apply personalized ranking if user_id provided
            if request.user_id and results:
                try:
                    results = personalized_ranking.personalize_results(request.user_id, results)
                except Exception as e:
                    logger.warning(f"Personalization failed: {e}")
            
            # Apply diversity if requested
            if request.diversify and results:
                try:
                    results = diversity_ranking.diversify_results(results)
                except Exception as e:
                    logger.warning(f"Diversification failed: {e}")
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
        
        # Apply pagination safely
        try:
            total_results = len(results) if results else 0
            start_idx = min(request.offset, total_results)
            end_idx = min(start_idx + request.limit, total_results)
            paginated_results = results[start_idx:end_idx] if results else []
        except Exception as e:
            logger.error(f"Pagination error: {e}")
            paginated_results = []
            total_results = 0
        
        # Convert to API models with error handling
        api_results = []
        try:
            for result in paginated_results:
                try:
                    pub_model = PublicationModel(
                        id=result.publication.id,
                        title=result.publication.title or "",
                        authors=result.publication.authors or [],
                        abstract=result.publication.abstract or "",
                        year=result.publication.year or "",
                        link=result.publication.link or ""
                    )
                    api_results.append(SearchResultModel(
                        publication=pub_model,
                        score=result.score,
                        matched_fields=result.matched_fields or []
                    ))
                except Exception as e:
                    logger.warning(f"Error converting result: {e}")
                    continue
        except Exception as e:
            logger.error(f"Results conversion error: {e}")
        
        # Get suggestions with error handling
        suggestions = []
        try:
            suggestions = engine.get_suggestions(request.query, limit=5)
        except Exception as e:
            logger.warning(f"Suggestions failed: {e}")
        
        # Generate facets with error handling
        facets = {}
        try:
            facets = generate_facets(results) if results else {}
        except Exception as e:
            logger.warning(f"Facets generation failed: {e}")
        
        query_time = time.time() - start_time
        
        return SearchResponse(
            results=api_results,
            total_results=total_results,
            query_time=query_time,
            query_info={
                "original_query": request.query,
                "parsed_query": validation['sanitized_query'] if 'validation' in locals() else request.query,
                "query_type": parsed_query.query_type.value if 'parsed_query' in locals() and hasattr(parsed_query, 'query_type') else "unknown",
                "term_count": len(parsed_query.terms) if 'parsed_query' in locals() and hasattr(parsed_query, 'terms') else 0,
                "expanded": request.expand_query
            },
            suggestions=suggestions,
            facets=facets
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        # Return structured error response instead of 500
        return SearchResponse(
            results=[],
            total_results=0,
            query_time=time.time() - start_time,
            query_info={
                "original_query": request.query,
                "error": "Search failed"
            },
            suggestions=[],
            facets={}
        )

@app.get("/search", response_model=SearchResponse)
async def search_publications_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    year: Optional[str] = Query(None, description="Filter by year"),
    author: Optional[str] = Query(None, description="Filter by author"),
    sort_by: str = Query(default="relevance", description="Sort by field"),
    expand: bool = Query(default=False, description="Expand query with synonyms"),
    engine: SearchEngine = Depends(get_search_engine)
):
    """Search endpoint via GET request"""
    filters = {}
    if year:
        filters['year'] = year
    if author:
        filters['author'] = author
    
    request = SearchRequest(
        query=q,
        limit=limit,
        offset=offset,
        filters=filters if filters else None,
        sort_by=sort_by,
        expand_query=expand
    )
    
    return await search_publications(request, engine)

@app.post("/suggestions", response_model=List[str])
async def get_suggestions(
    request: SuggestionsRequest,
    engine: SearchEngine = Depends(get_search_engine)
):
    """Get search suggestions"""
    try:
        suggestions = engine.get_suggestions(request.query, limit=request.limit)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions", response_model=List[str])
async def get_suggestions_get(
    q: str = Query(..., description="Partial query"),
    limit: int = Query(default=5, ge=1, le=10, description="Number of suggestions"),
    engine: SearchEngine = Depends(get_search_engine)
):
    """Get search suggestions via GET"""
    return engine.get_suggestions(q, limit=limit)

@app.get("/publication/{publication_id}", response_model=PublicationModel)
async def get_publication(
    publication_id: int,
    engine: SearchEngine = Depends(get_search_engine)
):
    """Get specific publication by ID"""
    if publication_id not in engine.index.publications:
        raise HTTPException(status_code=404, detail="Publication not found")
    
    pub = engine.index.publications[publication_id]
    return PublicationModel(
        id=pub.id,
        title=pub.title,
        authors=pub.authors,
        abstract=pub.abstract,
        year=pub.year,
        link=pub.link
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats(engine: SearchEngine = Depends(get_search_engine)):
    """Get search engine statistics"""
    stats = engine.get_stats()
    return StatsResponse(
        total_publications=stats['total_publications'],
        total_terms=stats['total_terms'],
        index_size=stats['index_size'],
        last_updated=datetime.now().isoformat()
    )

@app.post("/rebuild-index")
async def rebuild_index(engine: SearchEngine = Depends(get_search_engine)):
    """Rebuild search index (admin endpoint)"""
    try:
        engine.index.build_index()
        return {"message": "Index rebuilt successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

@app.post("/user/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    preferences: Dict
):
    """Update user preferences for personalized ranking"""
    try:
        personalized_ranking.update_user_preference(user_id, preferences)
        return {"message": "Preferences updated successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/{user_id}/search-feedback")
async def record_search_feedback(
    user_id: str,
    query: str,
    clicked_results: List[int]
):
    """Record user search feedback for learning"""
    try:
        personalized_ranking.record_search(user_id, query, clicked_results)
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facets")
async def get_facets(
    query: Optional[str] = Query(None, description="Query to generate facets for"),
    engine: SearchEngine = Depends(get_search_engine)
):
    """Get facet information for search refinement"""
    if query:
        results = engine.search(query, limit=1000)
        return generate_facets(results)
    else:
        # Return general facets from all publications
        return generate_general_facets(engine)

def generate_facets(results: List[SearchResult]) -> Dict[str, List[Dict]]:
    """Generate facets from search results"""
    year_counts = {}
    author_counts = {}
    
    for result in results:
        pub = result.publication
        
        # Year facets
        if pub.year:
            year_counts[pub.year] = year_counts.get(pub.year, 0) + 1
        
        # Author facets
        for author in pub.authors:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    # Sort and limit facets
    year_facets = [
        {"value": year, "count": count}
        for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True)[:10]
    ]
    
    author_facets = [
        {"value": author, "count": count}
        for author, count in sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    ]
    
    return {
        "years": year_facets,
        "authors": author_facets
    }

def generate_general_facets(engine: SearchEngine) -> Dict[str, List[Dict]]:
    """Generate general facets from all publications"""
    year_counts = {}
    author_counts = {}
    
    for pub in engine.index.publications.values():
        if pub.year:
            year_counts[pub.year] = year_counts.get(pub.year, 0) + 1
        
        for author in pub.authors:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    year_facets = [
        {"value": year, "count": count}
        for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True)[:20]
    ]
    
    author_facets = [
        {"value": author, "count": count}
        for author, count in sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    ]
    
    return {
        "years": year_facets,
        "authors": author_facets
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        search_healthy = check_search_health()
        circuit_status = circuit_breaker.state
        
        overall_status = "healthy" if search_healthy and circuit_status == "CLOSED" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "search_engine": "healthy" if search_healthy else "unhealthy",
                "circuit_breaker": circuit_status,
                "failure_count": circuit_breaker.failure_count
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "error": str(e)
        }

@app.get("/robots")
async def robots_status():
    """Get robots.txt compliance status and summary"""
    try:
        return get_robots_summary()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get robots.txt status: {str(e)}"
        }

@app.get("/search/test")
async def test_search():
    """Test endpoint to check if search functionality is working"""
    try:
        engine = await get_search_engine()
        
        # Perform a quick test search
        results = execute_search_with_timeout(
            engine.search,
            "test",
            limit=1,
            timeout_seconds=5
        )
        
        return {
            "status": "success",
            "search_engine_available": True,
            "index_loaded": hasattr(engine.index, 'publications') and len(engine.index.publications) > 0,
            "total_publications": len(engine.index.publications) if hasattr(engine.index, 'publications') else 0,
            "test_results_count": len(results) if results else 0
        }
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "search_engine_available": False
        }

# Crawler Management Endpoints
@app.post("/crawler/jobs", response_model=Dict[str, str])
async def create_crawl_job(request: CrawlJobRequest):
    """Create a new crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        frequency = CrawlFrequency(request.frequency)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid frequency value")
    
    try:
        job_id = crawler_scheduler.create_job(
            name=request.name,
            frequency=frequency,
            max_pages=request.max_pages,
            max_workers=request.max_workers,
            custom_schedule=request.custom_schedule
        )
        
        if not request.enabled:
            crawler_scheduler.disable_job(job_id)
        
        return {"job_id": job_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create crawl job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawler/jobs", response_model=List[CrawlJobResponse])
async def get_crawl_jobs():
    """Get all crawl jobs"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        jobs = crawler_scheduler.db.get_all_jobs()
        response_jobs = []
        
        for job in jobs:
            job_status = crawler_scheduler.get_job_status(job.id)
            response_jobs.append(CrawlJobResponse(
                id=job.id,
                name=job.name,
                frequency=job.frequency.value,
                enabled=job.enabled,
                last_run=job.last_run.isoformat() if job.last_run else None,
                next_run=job.next_run.isoformat() if job.next_run else None,
                status=job.status.value,
                max_pages=job.max_pages,
                max_workers=job.max_workers,
                custom_schedule=job.custom_schedule,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                run_count=job.run_count,
                success_count=job.success_count,
                failure_count=job.failure_count,
                is_running=job_status.get("is_running", False),
                success_rate=job_status.get("success_rate", 0.0)
            ))
        
        return response_jobs
    except Exception as e:
        logger.error(f"Failed to get crawl jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawler/jobs/{job_id}", response_model=CrawlJobResponse)
async def get_crawl_job(job_id: str):
    """Get a specific crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        job = crawler_scheduler.db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = crawler_scheduler.get_job_status(job_id)
        
        return CrawlJobResponse(
            id=job.id,
            name=job.name,
            frequency=job.frequency.value,
            enabled=job.enabled,
            last_run=job.last_run.isoformat() if job.last_run else None,
            next_run=job.next_run.isoformat() if job.next_run else None,
            status=job.status.value,
            max_pages=job.max_pages,
            max_workers=job.max_workers,
            custom_schedule=job.custom_schedule,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            run_count=job.run_count,
            success_count=job.success_count,
            failure_count=job.failure_count,
            is_running=job_status.get("is_running", False),
            success_rate=job_status.get("success_rate", 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/crawler/jobs/{job_id}/enable", response_model=Dict[str, str])
async def enable_crawl_job(job_id: str):
    """Enable a crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    if not crawler_scheduler.enable_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "status": "enabled"}

@app.put("/crawler/jobs/{job_id}/disable", response_model=Dict[str, str])
async def disable_crawl_job(job_id: str):
    """Disable a crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    if not crawler_scheduler.disable_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "status": "disabled"}

@app.post("/crawler/jobs/{job_id}/run", response_model=Dict[str, str])
async def run_crawl_job(job_id: str):
    """Run a crawl job immediately"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    if not crawler_scheduler.run_job_now(job_id):
        job = crawler_scheduler.db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=409, detail="Job is already running")
    
    return {"job_id": job_id, "status": "started"}

@app.delete("/crawler/jobs/{job_id}", response_model=Dict[str, str])
async def delete_crawl_job(job_id: str):
    """Delete a crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    if not crawler_scheduler.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "status": "deleted"}

@app.get("/crawler/status", response_model=Dict)
async def get_crawler_status():
    """Get crawler scheduler status"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        return crawler_scheduler.get_scheduler_status()
    except Exception as e:
        logger.error(f"Failed to get crawler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawler/jobs/{job_id}/results", response_model=List[Dict])
async def get_crawl_job_results(job_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """Get results for a specific crawl job"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        results = crawler_scheduler.db.get_job_results(job_id, limit)
        return [
            {
                "job_id": result.job_id,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "status": result.status.value,
                "publications_found": result.publications_found,
                "publications_added": result.publications_added,
                "publications_updated": result.publications_updated,
                "error_message": result.error_message,
                "execution_time": result.execution_time
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Failed to get job results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawler/jobs/create-defaults", response_model=Dict[str, List[str]])
async def create_default_crawl_jobs():
    """Create default crawl jobs"""
    global crawler_scheduler
    if not crawler_scheduler:
        raise HTTPException(status_code=503, detail="Crawler scheduler not available")
    
    try:
        job_ids = []
        
        # Daily full crawl
        job_id1 = crawler_scheduler.create_job(
            name="Daily Full Crawl",
            frequency=CrawlFrequency.DAILY,
            max_pages=None,
            max_workers=2
        )
        job_ids.append(job_id1)
        
        # Weekly comprehensive crawl
        job_id2 = crawler_scheduler.create_job(
            name="Weekly Comprehensive Crawl",
            frequency=CrawlFrequency.WEEKLY,
            max_pages=None,
            max_workers=3
        )
        job_ids.append(job_id2)
        
        # Hourly quick check
        job_id3 = crawler_scheduler.create_job(
            name="Hourly Quick Check",
            frequency=CrawlFrequency.HOURLY,
            max_pages=2,
            max_workers=1
        )
        job_ids.append(job_id3)
        
        return {"created_jobs": job_ids}
        
    except Exception as e:
        logger.error(f"Failed to create default jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Academic Search Engine API...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )