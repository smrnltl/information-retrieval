# Unified Academic Tools

A combined web application featuring an Academic Search Engine and Document Classifier with a modern, unified interface.

## Features

### 🔍 Academic Search Engine (Task-1)
- Search through academic publications and research papers
- Advanced query processing and ranking system
- Web crawler for academic content
- FastAPI-based search API

### 📋 Document Classifier (Task-2) 
- Classify documents into Politics, Business, or Health categories
- Machine learning model with 95.7% accuracy
- Trained on 114 real-world articles from international news portals
- Real-time classification with confidence scores

## Project Structure

```
solution-final/
├── ui/                     # Centralized UI Application
│   ├── main.py            # FastAPI main application
│   ├── launch.py          # Application launcher
│   ├── requirements.txt   # Combined dependencies
│   └── templates/
│       └── index.html     # Unified web interface
├── task-1/                # Academic Search Engine (Backend)
│   ├── search_engine.py   # Search engine core
│   ├── query_processor.py # Query processing
│   ├── ranking_system.py  # Result ranking
│   ├── crawler.py         # Web crawler
│   └── ...                # Other engine files
└── task-2/                # Document Classifier (Backend)
    ├── full_article_classifier.py  # ML classifier
    ├── *_articles.txt      # Training data
    └── ...                 # Other classifier files
```

## Installation & Usage

### Quick Start
1. Navigate to the `ui` directory:
   ```bash
   cd ui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python launch.py
   ```

4. Open your browser at `http://localhost:8000`

### Manual Start
Alternatively, you can start the server manually:
```bash
cd ui
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage Guide

### Academic Search
1. Click the "Search" tab
2. Enter your search query in the search box
3. Press Enter or click "Search"
4. Browse through the ranked results

### Document Classification
1. Click the "Classify" tab  
2. Either:
   - Type/paste your text in the text area
   - Select from predefined examples
3. Click "Classify Document"
4. View the predicted category, confidence score, and probability breakdown

## Technical Details

### Backend Architecture
- **Framework**: FastAPI with async support
- **Search Engine**: Custom academic publication search with TF-IDF ranking
- **Classifier**: Logistic Regression with TF-IDF features (10,000 features + bigrams)
- **Model Accuracy**: 95.7% on test dataset

### Frontend Technology
- **Styling**: TailwindCSS for modern, responsive design
- **Interactivity**: AlpineJS for reactive components
- **Charts**: Chart.js for probability visualizations
- **Icons**: Font Awesome

### API Endpoints
- `GET /` - Main web interface
- `POST /api/search` - Academic search functionality
- `POST /api/classify` - Document classification
- `GET /api/health` - Health check

## Model Information

### Document Classifier
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF with 10,000 features + bigrams
- **Training Data**: 114 full-length articles
- **Sources**: International, Nepal, and Asian news portals
- **Categories**: Politics, Business, Health
- **Test Accuracy**: 95.7% (22/23 correct predictions)

## Development

The application uses a modular architecture where:
- Task-specific engines remain in their original folders (`task-1/`, `task-2/`)
- UI logic is centralized in the `ui/` folder
- Engines are imported as modules by the main application

This design allows for:
- Easy maintenance of individual components
- Unified user experience
- Potential for cross-feature integrations
- Scalable architecture for future enhancements