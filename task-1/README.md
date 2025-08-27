# Academic Search Engine

A comprehensive vertical search engine for academic publications, comparable to Google Scholar, built specifically for searching publications from the Coventry University portal.

## 🎯 Features

### Core Search Capabilities
- **Full-text search** across titles, authors, abstracts, and years
- **TF-IDF scoring** with relevance ranking
- **Boolean search** (AND, OR, NOT operators)
- **Phrase search** with quoted terms
- **Field-specific search** (title:, author:, year:, abstract:)
- **Query expansion** with academic domain synonyms
- **Auto-complete** and search suggestions

### Advanced Features
- **Faceted search** with year and author filters
- **Result pagination** and sorting options
- **Personalized ranking** based on user preferences
- **Result diversification** to avoid similar publications
- **Real-time analytics** and search metrics
- **Citation formatting** for academic references

### User Interface
- **Responsive web portal** with modern UI
- **Advanced search panel** with filters
- **Real-time search suggestions**
- **Publication detail views**
- **Export functionality** for search results

### API & Analytics
- **RESTful API** with comprehensive endpoints
- **Search analytics** and user behavior tracking
- **Performance monitoring** with response time metrics
- **Click-through rate** analysis
- **Popular queries** and trend analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Portal    │    │   REST API      │    │  Search Engine  │
│   (Frontend)    │◄──►│   (FastAPI)     │◄──►│   (Core Logic)  │
│   Port 8080     │    │   Port 8000     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Analytics     │    │   Query         │    │  Search Index   │
│   Database      │    │   Processor     │    │  (Inverted)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Publications DB │
                                               │   (SQLite)      │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Your existing `publications.db` and `publications.json` files from the crawler

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the complete system:**
```bash
python run_system.py
```

This will:
- ✅ Check all requirements
- 🔨 Build the search index
- 🧪 Test search functionality
- 🚀 Start the API server (port 8000)
- 🌐 Start the web server (port 8080)

### Access Points
- **Web Interface:** http://localhost:8080
- **API Server:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## 🔍 Usage Examples

### Web Interface
1. Open http://localhost:8080
2. Enter search terms in the search box
3. Use advanced search for filters and boolean operators
4. Click on results to view full publications

### API Usage

**Simple Search:**
```bash
curl "http://localhost:8000/search?q=machine%20learning&limit=10"
```

**Advanced Search with Filters:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "economics AND education",
    "limit": 20,
    "filters": {"year": "2020", "author": "smith"},
    "expand_query": true
  }'
```

**Get Search Suggestions:**
```bash
curl "http://localhost:8000/suggestions?q=econom&limit=5"
```

### Search Query Syntax

**Basic Search:**
- `machine learning` - Find publications containing both terms
- `"machine learning"` - Exact phrase search

**Boolean Search:**
- `economics AND education` - Both terms must be present
- `finance OR accounting` - Either term can be present
- `education NOT assessment` - First term present, second absent

**Field-Specific Search:**
- `title:"higher education"` - Search only in titles
- `author:smith` - Search for author named Smith
- `year:2020` - Publications from 2020
- `abstract:"artificial intelligence"` - Search in abstracts

**Combined Search:**
```
title:"machine learning" AND author:johnson year:2020
```

## 📊 Analytics & Monitoring

### Built-in Analytics
The system tracks:
- Search queries and response times
- Click-through rates by position
- Popular queries and publications
- User behavior patterns
- System performance metrics

### Accessing Analytics
```python
from analytics import AnalyticsReporter

reporter = AnalyticsReporter()
stats = reporter.get_search_stats(days=30)
print(stats)
```

### API Endpoints for Analytics
- `GET /stats` - General system statistics
- `POST /user/{user_id}/preferences` - Set user preferences
- `POST /user/{user_id}/search-feedback` - Record search feedback

## 🛠️ System Components

### 1. Search Engine (`search_engine.py`)
- **SearchIndex**: Builds and manages inverted indexes
- **SearchEngine**: Executes searches with TF-IDF scoring
- **Publication**: Data model for publications

### 2. Query Processor (`query_processor.py`)
- **QueryParser**: Parses different query types
- **QueryExpander**: Adds synonyms and related terms
- **QueryValidator**: Validates and sanitizes queries

### 3. Ranking System (`ranking_system.py`)
- **RankingSystem**: Comprehensive relevance scoring
- **PersonalizedRanking**: User-based personalization
- **DiversityRanking**: Result diversification

### 4. API Server (`api_server.py`)
- **FastAPI**: REST API with automatic documentation
- **Pydantic Models**: Request/response validation
- **CORS Support**: Cross-origin requests enabled

### 5. Analytics (`analytics.py`)
- **AnalyticsTracker**: Tracks search events and clicks
- **AnalyticsReporter**: Generates analytics reports
- **AnalyticsExporter**: Exports data to JSON

### 6. Web Interface (`templates/index.html`)
- **Alpine.js**: Reactive frontend framework
- **Tailwind CSS**: Modern styling
- **Responsive Design**: Works on all devices

## 🔧 Configuration

### Customizing Search Weights
Edit weights in `ranking_system.py`:
```python
self.weights = {
    'tf_idf': 0.3,          # TF-IDF relevance
    'title_match': 0.25,    # Title matches
    'author_match': 0.15,   # Author matches
    'abstract_match': 0.1,  # Abstract matches
    'year_relevance': 0.05, # Recent publications
    'citation_score': 0.05, # Citation-based score
    'query_coverage': 0.15, # Query term coverage
    'field_specificity': 0.1 # Field-specific bonuses
}
```

### Adding Synonyms
Edit synonyms in `query_processor.py`:
```python
self.synonyms = {
    'education': ['learning', 'teaching', 'academic'],
    'student': ['learner', 'pupil', 'scholar'],
    # Add more synonyms...
}
```

## 📈 Performance Optimization

### Index Optimization
- Indexes are built once and cached to disk
- Incremental updates supported for new publications
- Memory-efficient inverted index structure

### Search Optimization
- TF-IDF calculations cached per query
- Result pagination to limit memory usage
- Query result caching for popular searches

### API Optimization
- Async FastAPI for concurrent requests
- Response compression enabled
- Request validation with Pydantic

## 🧪 Testing

### Run Individual Components
```bash
# Test search engine
python search_engine.py

# Test query processor
python query_processor.py

# Test ranking system
python ranking_system.py

# Test analytics
python analytics.py
```

### API Testing
Use the interactive API documentation at http://localhost:8000/docs

## 📁 File Structure

```
├── search_engine.py      # Core search functionality
├── query_processor.py    # Query parsing and processing
├── ranking_system.py     # Relevance ranking algorithms
├── api_server.py         # REST API server
├── web_server.py         # Web interface server
├── analytics.py          # Search analytics and tracking
├── run_system.py         # System launcher script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface template
├── publications.db      # Your publication database
├── publications.json    # JSON export of publications
├── search_index.pkl     # Generated search index
└── analytics.db         # Analytics database
```

## 🚨 Troubleshooting

### Common Issues

**"No module named 'fastapi'"**
```bash
pip install -r requirements.txt
```

**"publications.db not found"**
- Run your crawler first to generate the database
- Ensure the database file is in the project directory

**"Search returns no results"**
- Check if the search index was built successfully
- Verify publications.db contains data
- Try rebuilding the index: `python -c "from search_engine import rebuild_index; rebuild_index()"`

**API server won't start**
- Check if port 8000 is already in use
- Look for error messages in the console
- Ensure all dependencies are installed

### Performance Issues

**Slow search responses**
- Check if the search index is built
- Monitor system resources during search
- Consider reducing result limits for large datasets

**High memory usage**
- Limit search result count
- Implement result pagination
- Monitor index size and optimize if needed

## 🔮 Future Enhancements

### Planned Features
- [ ] **Machine Learning Ranking**: ML-based relevance scoring
- [ ] **Real-time Indexing**: Live updates from crawler
- [ ] **Export Formats**: BibTeX, RIS, EndNote support
- [ ] **Search History**: User search history and bookmarks
- [ ] **Advanced Analytics**: Detailed user behavior analysis
- [ ] **Multi-language Support**: International publication support
- [ ] **Citation Networks**: Publication citation relationships
- [ ] **Recommendation Engine**: "Similar publications" feature

### Scalability Improvements
- [ ] **Elasticsearch Integration**: For large-scale deployments
- [ ] **Redis Caching**: Query result caching
- [ ] **Database Optimization**: PostgreSQL for large datasets
- [ ] **Load Balancing**: Multiple API server instances
- [ ] **CDN Integration**: Static asset optimization

## 📝 License

This project is developed for academic purposes. Please ensure compliance with the terms of service of the data sources.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Create an issue with detailed error information

---

**🎓 Academic Search Engine - Making research discoverable, one query at a time.**