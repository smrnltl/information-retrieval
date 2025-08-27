import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
import re

from search_engine import Publication, SearchResult
from query_processor import ParsedQuery, QueryTerm

@dataclass
class RankingFeatures:
    tf_idf_score: float
    title_match: float
    author_match: float
    abstract_match: float
    year_relevance: float
    citation_score: float
    query_coverage: float
    field_specificity: float

class RankingSystem:
    def __init__(self):
        # Weights for different ranking factors
        self.weights = {
            'tf_idf': 0.3,
            'title_match': 0.25,
            'author_match': 0.15,
            'abstract_match': 0.1,
            'year_relevance': 0.05,
            'citation_score': 0.05,
            'query_coverage': 0.15,
            'field_specificity': 0.1
        }
        
        self.current_year = datetime.now().year
    
    def calculate_ranking_score(self, publication: Publication, query: ParsedQuery, 
                              tf_idf_score: float, matched_fields: List[str]) -> Tuple[float, RankingFeatures]:
        """Calculate comprehensive ranking score for a publication"""
        
        features = RankingFeatures(
            tf_idf_score=tf_idf_score,
            title_match=self._calculate_title_match(publication, query),
            author_match=self._calculate_author_match(publication, query),
            abstract_match=self._calculate_abstract_match(publication, query),
            year_relevance=self._calculate_year_relevance(publication),
            citation_score=self._calculate_citation_score(publication),
            query_coverage=self._calculate_query_coverage(publication, query),
            field_specificity=self._calculate_field_specificity(query, matched_fields)
        )
        
        # Calculate weighted final score
        final_score = (
            features.tf_idf_score * self.weights['tf_idf'] +
            features.title_match * self.weights['title_match'] +
            features.author_match * self.weights['author_match'] +
            features.abstract_match * self.weights['abstract_match'] +
            features.year_relevance * self.weights['year_relevance'] +
            features.citation_score * self.weights['citation_score'] +
            features.query_coverage * self.weights['query_coverage'] +
            features.field_specificity * self.weights['field_specificity']
        )
        
        return final_score, features
    
    def _calculate_title_match(self, publication: Publication, query: ParsedQuery) -> float:
        """Calculate title matching score"""
        if not publication.title:
            return 0.0
        
        title_lower = publication.title.lower()
        score = 0.0
        
        for term in query.terms:
            if term.field and term.field != 'title':
                continue
            
            if term.is_phrase:
                # Exact phrase match in title
                if term.term in title_lower:
                    score += 2.0
            else:
                # Individual word match
                if term.term in title_lower:
                    score += 1.0
                
                # Bonus for word at beginning of title
                if title_lower.startswith(term.term):
                    score += 0.5
        
        # Normalize by query length
        return min(score / max(len(query.terms), 1), 1.0)
    
    def _calculate_author_match(self, publication: Publication, query: ParsedQuery) -> float:
        """Calculate author matching score"""
        if not publication.authors:
            return 0.0
        
        authors_text = " ".join(publication.authors).lower()
        score = 0.0
        
        for term in query.terms:
            if term.field and term.field not in ['authors', 'author']:
                continue
            
            if term.is_phrase:
                if term.term in authors_text:
                    score += 2.0
            else:
                if term.term in authors_text:
                    score += 1.0
                
                # Check for exact author name matches
                for author in publication.authors:
                    if term.term in author.lower():
                        score += 1.5
        
        return min(score / max(len(query.terms), 1), 1.0)
    
    def _calculate_abstract_match(self, publication: Publication, query: ParsedQuery) -> float:
        """Calculate abstract matching score"""
        if not publication.abstract:
            return 0.0
        
        abstract_lower = publication.abstract.lower()
        score = 0.0
        
        for term in query.terms:
            if term.field and term.field != 'abstract':
                continue
            
            if term.is_phrase:
                if term.term in abstract_lower:
                    score += 1.0
            else:
                # Count term frequency in abstract
                term_count = abstract_lower.count(term.term)
                score += min(term_count * 0.5, 2.0)  # Cap individual term contribution
        
        # Normalize by abstract length and query terms
        abstract_length = len(publication.abstract.split())
        normalization_factor = math.log(abstract_length + 1) * max(len(query.terms), 1)
        
        return min(score / normalization_factor, 1.0)
    
    def _calculate_year_relevance(self, publication: Publication) -> float:
        """Calculate year relevance (prefer recent publications)"""
        if not publication.year or not publication.year.isdigit():
            return 0.5  # Neutral score for unknown year
        
        pub_year = int(publication.year)
        
        # Publications from current year get max score
        if pub_year == self.current_year:
            return 1.0
        
        # Decay score for older publications
        year_diff = self.current_year - pub_year
        
        if year_diff <= 2:
            return 0.9
        elif year_diff <= 5:
            return 0.8
        elif year_diff <= 10:
            return 0.6
        elif year_diff <= 20:
            return 0.4
        else:
            return 0.2
    
    def _calculate_citation_score(self, publication: Publication) -> float:
        """Calculate citation-based score (placeholder - would need citation data)"""
        # This is a placeholder - in a real system, you would have citation counts
        # For now, we'll use a simple heuristic based on title length and abstract quality
        
        score = 0.5  # Base score
        
        # Longer titles often indicate more specific research
        if publication.title:
            title_words = len(publication.title.split())
            if 5 <= title_words <= 15:  # Optimal title length
                score += 0.2
        
        # Publications with abstracts likely have more citations
        if publication.abstract and len(publication.abstract) > 100:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_query_coverage(self, publication: Publication, query: ParsedQuery) -> float:
        """Calculate how well the publication covers the query terms"""
        if not query.terms:
            return 0.0
        
        covered_terms = 0
        total_terms = len([t for t in query.terms if not t.is_negated])
        
        pub_text = f"{publication.title} {' '.join(publication.authors)} {publication.abstract}".lower()
        
        for term in query.terms:
            if term.is_negated:
                continue
            
            if term.is_phrase:
                if term.term in pub_text:
                    covered_terms += 1
            else:
                if term.term in pub_text:
                    covered_terms += 1
        
        return covered_terms / max(total_terms, 1)
    
    def _calculate_field_specificity(self, query: ParsedQuery, matched_fields: List[str]) -> float:
        """Calculate bonus for field-specific matches"""
        if not matched_fields:
            return 0.0
        
        # Bonus points for different types of matches
        field_scores = {
            'title': 1.0,
            'authors': 0.8,
            'abstract': 0.6,
            'year': 0.4
        }
        
        score = 0.0
        for field in matched_fields:
            score += field_scores.get(field, 0.3)
        
        # Normalize by number of matched fields
        return min(score / len(matched_fields), 1.0)

class PersonalizedRanking:
    """Personalized ranking based on user preferences and search history"""
    
    def __init__(self):
        self.user_preferences = {}
        self.search_history = defaultdict(list)
    
    def update_user_preference(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        self.user_preferences[user_id] = preferences
    
    def record_search(self, user_id: str, query: str, clicked_results: List[int]):
        """Record user search and clicks for learning"""
        self.search_history[user_id].append({
            'query': query,
            'clicked_results': clicked_results,
            'timestamp': datetime.now()
        })
    
    def personalize_results(self, user_id: str, results: List[SearchResult]) -> List[SearchResult]:
        """Apply personalized ranking to results"""
        if user_id not in self.user_preferences:
            return results
        
        preferences = self.user_preferences[user_id]
        
        # Apply preference-based reranking
        for result in results:
            publication = result.publication
            
            # Boost based on preferred years
            if 'preferred_years' in preferences:
                if publication.year in preferences['preferred_years']:
                    result.score *= 1.2
            
            # Boost based on preferred authors
            if 'preferred_authors' in preferences:
                for author in publication.authors:
                    if any(pref_author.lower() in author.lower() 
                          for pref_author in preferences['preferred_authors']):
                        result.score *= 1.15
                        break
            
            # Boost based on preferred topics (keywords in title/abstract)
            if 'preferred_topics' in preferences:
                pub_text = f"{publication.title} {publication.abstract}".lower()
                for topic in preferences['preferred_topics']:
                    if topic.lower() in pub_text:
                        result.score *= 1.1
                        break
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class DiversityRanking:
    """Ensure diversity in search results"""
    
    def __init__(self):
        self.diversity_threshold = 0.7  # Similarity threshold for diversification
    
    def diversify_results(self, results: List[SearchResult], max_similar: int = 3) -> List[SearchResult]:
        """Apply result diversification to avoid too many similar results"""
        if len(results) <= max_similar:
            return results
        
        diversified = []
        
        for result in results:
            # Check similarity with already selected results
            similar_count = 0
            
            for selected in diversified:
                similarity = self._calculate_similarity(result.publication, selected.publication)
                if similarity > self.diversity_threshold:
                    similar_count += 1
            
            # Add result if it doesn't exceed similarity limit
            if similar_count < max_similar:
                diversified.append(result)
            
            # Stop if we have enough diverse results
            if len(diversified) >= len(results) * 0.8:  # Keep 80% of original results
                break
        
        # Fill remaining slots with highest-scoring results
        remaining_slots = len(results) - len(diversified)
        if remaining_slots > 0:
            remaining_results = [r for r in results if r not in diversified]
            diversified.extend(remaining_results[:remaining_slots])
        
        return diversified
    
    def _calculate_similarity(self, pub1: Publication, pub2: Publication) -> float:
        """Calculate similarity between two publications"""
        # Simple similarity based on shared words in title and authors
        
        # Title similarity
        title1_words = set(pub1.title.lower().split()) if pub1.title else set()
        title2_words = set(pub2.title.lower().split()) if pub2.title else set()
        
        title_similarity = 0.0
        if title1_words and title2_words:
            title_similarity = len(title1_words & title2_words) / len(title1_words | title2_words)
        
        # Author similarity
        authors1 = set(author.lower() for author in pub1.authors)
        authors2 = set(author.lower() for author in pub2.authors)
        
        author_similarity = 0.0
        if authors1 and authors2:
            author_similarity = len(authors1 & authors2) / len(authors1 | authors2)
        
        # Year similarity (exact match)
        year_similarity = 1.0 if pub1.year == pub2.year else 0.0
        
        # Weighted combination
        return (title_similarity * 0.6 + author_similarity * 0.3 + year_similarity * 0.1)

if __name__ == "__main__":
    # Test ranking system
    from search_engine import SearchEngine
    from query_processor import QueryParser
    
    # Create test data
    engine = SearchEngine()
    parser = QueryParser()
    ranking = RankingSystem()
    
    # Test query
    query = parser.parse("higher education economics")
    
    print(f"Testing ranking system with query: 'higher education economics'")
    print(f"Ranking weights: {ranking.weights}")
    print(f"\nFeature calculation methods available for comprehensive scoring.")