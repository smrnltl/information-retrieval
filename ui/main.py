import sys
import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json

# Add parent directory to path to import from task folders
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "task-1"))
sys.path.insert(0, str(parent_dir / "task-2"))

# Import engines from task folders  
SearchEngine = None
QueryProcessor = None
RankingSystem = None
FullArticleClassifier = None

def import_engines():
    """Import engines with proper path handling"""
    global SearchEngine, QueryProcessor, RankingSystem, FullArticleClassifier
    
    try:
        # Change to task-1 directory and import search modules
        original_dir = os.getcwd()
        task1_dir = parent_dir / "task-1"
        os.chdir(str(task1_dir))
        
        import search_engine
        import query_processor 
        import ranking_system
        
        SearchEngine = search_engine.SearchEngine
        QueryProcessor = query_processor.QueryProcessor
        RankingSystem = ranking_system.RankingSystem
        
        os.chdir(original_dir)
        print("Task-1 modules imported successfully")
    except Exception as e:
        print(f"Warning: Task-1 modules import failed: {e}")
        try:
            os.chdir(original_dir)
        except:
            pass

    try:
        # Change to task-2 directory and import classifier
        original_dir = os.getcwd()
        task2_dir = parent_dir / "task-2"
        os.chdir(str(task2_dir))
        
        import full_article_classifier
        FullArticleClassifier = full_article_classifier.FullArticleClassifier
        
        os.chdir(original_dir)
        print("Task-2 modules imported successfully")
    except Exception as e:
        print(f"Warning: Task-2 modules import failed: {e}")
        try:
            os.chdir(original_dir)
        except:
            pass

# Import engines at module level
import_engines()

# Create FastAPI app
app = FastAPI(
    title="Unified Academic Tools",
    description="Combined Academic Search Engine and Document Classifier",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory=str(current_dir / "templates"))

# Create static directory if it doesn't exist
static_dir = current_dir / "static"
if not static_dir.exists():
    static_dir.mkdir()

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize engines
search_engine = None
classifier = None

@app.on_event("startup") 
async def startup_event():
    """Initialize engines on startup"""
    global search_engine, classifier
    
    # Initialize search engine
    if SearchEngine:
        try:
            # Change to task-1 directory to find the database
            original_dir = os.getcwd()
            task1_dir = parent_dir / "task-1"
            os.chdir(str(task1_dir))
            
            search_engine = SearchEngine()
            print("Search Engine initialized")
            
            # Change back to original directory
            os.chdir(original_dir)
        except Exception as e:
            print(f"Failed to initialize Search Engine: {e}")
            # Ensure we change back to original directory even on error
            try:
                os.chdir(original_dir)
            except:
                pass
    
    # Initialize classifier
    if FullArticleClassifier:
        try:
            # Change to task-2 directory to find the data files
            original_dir = os.getcwd()
            task2_dir = parent_dir / "task-2"
            os.chdir(str(task2_dir))
            
            classifier = FullArticleClassifier()
            classifier.train()
            print("Document Classifier initialized")
            
            # Change back to original directory
            os.chdir(original_dir)
        except Exception as e:
            print(f"Failed to initialize Document Classifier: {e}")
            # Ensure we change back to original directory even on error
            try:
                os.chdir(original_dir)
            except:
                pass

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    num_results: int = 10

class ClassifyRequest(BaseModel):
    text: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """Search academic documents"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    try:
        import time
        start_time = time.time()
        
        # Change to task-1 directory for search operations
        original_dir = os.getcwd()
        task1_dir = parent_dir / "task-1"
        os.chdir(str(task1_dir))
        
        # Get more results than requested to show total count
        max_results = max(50, request.num_results * 2)  # Get more for accurate count
        all_results = search_engine.search(request.query, limit=max_results)
        
        # Slice to requested number
        results = all_results[:request.num_results]
        
        # Change back to original directory
        os.chdir(original_dir)
        
        search_time = round(time.time() - start_time, 2)
        
        # Convert results to JSON-serializable format
        formatted_results = []
        for result in results:
            # Truncate abstract to 200 characters
            abstract = result.publication.abstract or ""
            if len(abstract) > 200:
                abstract = abstract[:200].rsplit(' ', 1)[0] + "..."
            
            # Convert score to reasonable relevancy percentage (normalize to 0-100%)
            relevancy_percent = min(100, max(0, (result.score / 10)))  # Adjust scale as needed
            
            # Parse authors if they're stored as JSON string
            authors = result.publication.authors
            if isinstance(authors, str):
                try:
                    import json
                    if authors.startswith('['):
                        authors = json.loads(authors)
                        authors = ', '.join(authors) if isinstance(authors, list) else authors
                except:
                    pass  # Keep as string if parsing fails
            elif isinstance(authors, list):
                authors = ', '.join(authors)
            
            formatted_results.append({
                "id": result.publication.id,
                "title": result.publication.title,
                "authors": authors,
                "abstract": abstract,
                "year": result.publication.year,
                "link": result.publication.link,
                "score": result.score,
                "relevancy": round(relevancy_percent, 1),
                "matched_fields": result.matched_fields
            })
        
        return {
            "results": formatted_results, 
            "query": request.query,
            "total_results": len(all_results),  # Total available results
            "showing_results": len(formatted_results),  # Currently displayed
            "search_time": search_time
        }
    except Exception as e:
        # Ensure we change back to original directory even on error
        try:
            os.chdir(original_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/classify")
async def classify_document(request: ClassifyRequest):
    """Classify document text"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    try:
        # Preprocess and classify
        preprocessed_text = classifier.preprocess_text(request.text)
        text_vec = classifier.vectorizer.transform([preprocessed_text])
        prediction_encoded = classifier.model.predict(text_vec)[0]
        prediction = classifier.label_encoder.inverse_transform([prediction_encoded])[0]
        probabilities = classifier.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        # Format response
        class_names = classifier.label_encoder.classes_.tolist()
        prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "search_engine": search_engine is not None,
        "classifier": classifier is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Unified Academic Tools...")
    print("Academic Search Engine + Document Classifier")
    print("Access the interface at: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )