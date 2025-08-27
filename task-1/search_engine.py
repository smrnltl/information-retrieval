import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import math
from dataclasses import dataclass
import pickle
import os

@dataclass
class Publication:
    id: int
    title: str
    authors: List[str]
    abstract: str
    year: str
    link: str

@dataclass
class SearchResult:
    publication: Publication
    score: float
    matched_fields: List[str]

class SearchIndex:
    def __init__(self, db_path: str = "publications.db"):
        self.db_path = db_path
        self.index_file = "search_index.pkl"
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        self.publications = {}
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def build_index(self):
        """Build inverted index from publications database"""
        print("Building search index...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, title, authors, abstract, year, link FROM publications")
        rows = cursor.fetchall()
        conn.close()
        
        self.total_documents = len(rows)
        print(f"Indexing {self.total_documents} publications...")
        
        for row in rows:
            pub_id, title, authors_json, abstract, year, link = row
            
            # Parse authors JSON
            try:
                authors = json.loads(authors_json) if authors_json else []
            except:
                authors = []
            
            # Create publication object
            pub = Publication(
                id=pub_id,
                title=title or "",
                authors=authors,
                abstract=abstract or "",
                year=year or "",
                link=link or ""
            )
            self.publications[pub_id] = pub
            
            # Index title
            title_tokens = self.tokenize(title)
            for token in title_tokens:
                self.inverted_index[token]['title'].append(pub_id)
            
            # Index authors
            authors_text = " ".join(authors)
            author_tokens = self.tokenize(authors_text)
            for token in author_tokens:
                self.inverted_index[token]['authors'].append(pub_id)
            
            # Index abstract
            abstract_tokens = self.tokenize(abstract)
            for token in abstract_tokens:
                self.inverted_index[token]['abstract'].append(pub_id)
            
            # Index year
            if year:
                self.inverted_index[year]['year'].append(pub_id)
        
        # Calculate document frequencies
        for term in self.inverted_index:
            doc_set = set()
            for field in self.inverted_index[term]:
                doc_set.update(self.inverted_index[term][field])
            self.document_frequencies[term] = len(doc_set)
        
        print(f"Index built with {len(self.inverted_index)} unique terms")
        self.save_index()
    
    def save_index(self):
        """Save index to disk"""
        index_data = {
            'inverted_index': dict(self.inverted_index),
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents,
            'publications': self.publications
        }
        
        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {self.index_file}")
    
    def load_index(self):
        """Load index from disk"""
        if not os.path.exists(self.index_file):
            print("No existing index found. Building new index...")
            self.build_index()
            return
        
        try:
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.inverted_index = defaultdict(lambda: defaultdict(list), index_data['inverted_index'])
            self.document_frequencies = defaultdict(int, index_data['document_frequencies'])
            self.total_documents = index_data['total_documents']
            self.publications = index_data['publications']
            
            print(f"Index loaded with {len(self.inverted_index)} terms and {self.total_documents} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Building new index...")
            self.build_index()

class SearchEngine:
    def __init__(self, db_path: str = "publications.db"):
        self.index = SearchIndex(db_path)
        self.index.load_index()
    
    def calculate_tf_idf(self, term: str, field: str, doc_id: int) -> float:
        """Calculate TF-IDF score for a term in a document field"""
        if term not in self.index.inverted_index:
            return 0.0
        
        if field not in self.index.inverted_index[term]:
            return 0.0
        
        if doc_id not in self.index.inverted_index[term][field]:
            return 0.0
        
        # Term frequency (count of term in field)
        tf = self.index.inverted_index[term][field].count(doc_id)
        
        # Document frequency
        df = self.index.document_frequencies[term]
        
        # TF-IDF calculation
        if df == 0:
            return 0.0
        
        idf = math.log(self.index.total_documents / df)
        return tf * idf
    
    def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for publications matching the query"""
        try:
            # Input validation
            if not query or not query.strip():
                return []
            
            # Ensure index is loaded
            if not hasattr(self.index, 'publications') or not self.index.publications:
                print("Warning: Search index not loaded properly")
                return []
            
            query_terms = self.index.tokenize(query)
            if not query_terms:
                return []
            
            # Find candidate documents with timeout protection
            candidate_docs = set()
            try:
                for term in query_terms:
                    if term in self.index.inverted_index:
                        for field in self.index.inverted_index[term]:
                            candidate_docs.update(self.index.inverted_index[term][field])
            except Exception as e:
                print(f"Error finding candidate documents: {e}")
                return []
            
            if not candidate_docs:
                return []
            
            # Score documents with error handling
            doc_scores = {}
            try:
                for doc_id in candidate_docs:
                    if doc_id not in self.index.publications:
                        continue
                    
                    score = 0.0
                    matched_fields = []
                    
                    try:
                        for term in query_terms:
                            # Title matches get higher weight
                            title_score = self.calculate_tf_idf(term, 'title', doc_id) * 3.0
                            if title_score > 0:
                                score += title_score
                                if 'title' not in matched_fields:
                                    matched_fields.append('title')
                            
                            # Author matches
                            author_score = self.calculate_tf_idf(term, 'authors', doc_id) * 2.0
                            if author_score > 0:
                                score += author_score
                                if 'authors' not in matched_fields:
                                    matched_fields.append('authors')
                            
                            # Abstract matches
                            abstract_score = self.calculate_tf_idf(term, 'abstract', doc_id)
                            if abstract_score > 0:
                                score += abstract_score
                                if 'abstract' not in matched_fields:
                                    matched_fields.append('abstract')
                            
                            # Year matches
                            year_score = self.calculate_tf_idf(term, 'year', doc_id) * 1.5
                            if year_score > 0:
                                score += year_score
                                if 'year' not in matched_fields:
                                    matched_fields.append('year')
                        
                        if score > 0:
                            doc_scores[doc_id] = (score, matched_fields)
                    except Exception as e:
                        print(f"Error scoring document {doc_id}: {e}")
                        continue
            except Exception as e:
                print(f"Error in document scoring: {e}")
                return []
            
            # Apply filters with error handling
            if filters:
                try:
                    filtered_scores = {}
                    for doc_id, (score, matched_fields) in doc_scores.items():
                        try:
                            pub = self.index.publications.get(doc_id)
                            if not pub:
                                continue
                            
                            # Year filter
                            if 'year' in filters and filters['year']:
                                if pub.year != str(filters['year']):
                                    continue
                            
                            # Author filter
                            if 'author' in filters and filters['author']:
                                author_filter = filters['author'].lower()
                                if pub.authors:
                                    author_match = any(author_filter in author.lower() 
                                                     for author in pub.authors if author)
                                    if not author_match:
                                        continue
                                else:
                                    continue
                            
                            filtered_scores[doc_id] = (score, matched_fields)
                        except Exception as e:
                            print(f"Error applying filters to document {doc_id}: {e}")
                            continue
                    
                    doc_scores = filtered_scores
                except Exception as e:
                    print(f"Error applying filters: {e}")
                    # Continue with unfiltered results
            
            # Sort by score with error handling
            try:
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)
            except Exception as e:
                print(f"Error sorting results: {e}")
                return []
            
            # Create search results with error handling
            results = []
            try:
                for doc_id, (score, matched_fields) in sorted_docs[:limit]:
                    try:
                        pub = self.index.publications.get(doc_id)
                        if pub:
                            results.append(SearchResult(
                                publication=pub,
                                score=score,
                                matched_fields=matched_fields
                            ))
                    except Exception as e:
                        print(f"Error creating result for document {doc_id}: {e}")
                        continue
            except Exception as e:
                print(f"Error creating search results: {e}")
                return []
            
            return results
            
        except Exception as e:
            print(f"Unexpected error in search: {e}")
            return []
    
    def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        if len(query) < 2:
            return []
        
        query_lower = query.lower()
        suggestions = []
        
        for term in self.index.inverted_index:
            if term.startswith(query_lower):
                suggestions.append(term)
        
        return sorted(suggestions)[:limit]
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return {
            'total_publications': self.index.total_documents,
            'total_terms': len(self.index.inverted_index),
            'index_size': os.path.getsize(self.index.index_file) if os.path.exists(self.index.index_file) else 0
        }

def rebuild_index():
    """Utility function to rebuild the search index"""
    engine = SearchEngine()
    engine.index.build_index()
    print("Search index rebuilt successfully!")

if __name__ == "__main__":
    # Test the search engine
    engine = SearchEngine()
    
    # Test search
    results = engine.search("higher education", limit=5)
    print(f"\nSearch results for 'higher education':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.publication.title[:60]}...")
        print(f"   Authors: {', '.join(result.publication.authors[:3])}")
        print(f"   Year: {result.publication.year}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Matched fields: {', '.join(result.matched_fields)}")
        print()