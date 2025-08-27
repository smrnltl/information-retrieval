import re
from typing import List, Dict, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    SIMPLE = "simple"
    PHRASE = "phrase"
    BOOLEAN = "boolean"
    ADVANCED = "advanced"

@dataclass
class QueryTerm:
    term: str
    field: Optional[str] = None
    operator: str = "AND"
    is_phrase: bool = False
    is_negated: bool = False

@dataclass
class ParsedQuery:
    terms: List[QueryTerm]
    query_type: QueryType
    filters: Dict[str, Union[str, int, List[str]]] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"

class QueryParser:
    def __init__(self):
        self.field_mapping = {
            'title': 'title',
            'author': 'authors',
            'authors': 'authors',
            'abstract': 'abstract',
            'year': 'year',
            'date': 'year'
        }
    
    def parse(self, query_string: str, filters: Optional[Dict] = None) -> ParsedQuery:
        """Parse query string into structured query"""
        if not query_string.strip():
            return ParsedQuery(terms=[], query_type=QueryType.SIMPLE, filters=filters or {})
        
        # Clean the query
        query = query_string.strip()
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Parse based on type
        if query_type == QueryType.BOOLEAN:
            terms = self._parse_boolean_query(query)
        elif query_type == QueryType.PHRASE:
            terms = self._parse_phrase_query(query)
        elif query_type == QueryType.ADVANCED:
            terms = self._parse_advanced_query(query)
        else:
            terms = self._parse_simple_query(query)
        
        return ParsedQuery(
            terms=terms,
            query_type=query_type,
            filters=filters or {},
            sort_by="relevance",
            sort_order="desc"
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        # Check for phrase query (quoted text)
        if '"' in query and query.count('"') >= 2:
            return QueryType.PHRASE
        
        # Check for field-specific queries
        field_pattern = r'\b(?:title|author|authors|abstract|year|date):'
        if re.search(field_pattern, query, re.IGNORECASE):
            return QueryType.ADVANCED
        
        # Check for boolean operators
        boolean_pattern = r'\b(?:AND|OR|NOT)\b'
        if re.search(boolean_pattern, query, re.IGNORECASE):
            return QueryType.BOOLEAN
        
        return QueryType.SIMPLE
    
    def _parse_simple_query(self, query: str) -> List[QueryTerm]:
        """Parse simple keyword query"""
        words = query.split()
        terms = []
        
        for word in words:
            if word.strip():
                terms.append(QueryTerm(term=word.lower().strip(), operator="AND"))
        
        return terms
    
    def _parse_phrase_query(self, query: str) -> List[QueryTerm]:
        """Parse phrase queries with quoted text"""
        terms = []
        
        # Find quoted phrases
        phrase_pattern = r'"([^"]+)"'
        phrases = re.findall(phrase_pattern, query)
        
        # Remove phrases from query to get remaining words
        remaining_query = re.sub(phrase_pattern, '', query)
        
        # Add phrase terms
        for phrase in phrases:
            terms.append(QueryTerm(term=phrase.lower().strip(), is_phrase=True, operator="AND"))
        
        # Add remaining individual words
        remaining_words = remaining_query.split()
        for word in remaining_words:
            if word.strip():
                terms.append(QueryTerm(term=word.lower().strip(), operator="AND"))
        
        return terms
    
    def _parse_boolean_query(self, query: str) -> List[QueryTerm]:
        """Parse boolean queries with AND, OR, NOT operators"""
        terms = []
        
        # Split by boolean operators while preserving them
        parts = re.split(r'\b(AND|OR|NOT)\b', query, flags=re.IGNORECASE)
        
        current_operator = "AND"
        for i, part in enumerate(parts):
            part = part.strip()
            
            if part.upper() in ['AND', 'OR', 'NOT']:
                current_operator = part.upper()
            elif part:
                # Handle quoted phrases within boolean queries
                if '"' in part:
                    phrase_matches = re.findall(r'"([^"]+)"', part)
                    for phrase in phrase_matches:
                        is_negated = current_operator == "NOT"
                        terms.append(QueryTerm(
                            term=phrase.lower().strip(),
                            is_phrase=True,
                            operator=current_operator if not is_negated else "AND",
                            is_negated=is_negated
                        ))
                    
                    # Remove phrases and process remaining words
                    remaining = re.sub(r'"[^"]+"', '', part)
                    words = remaining.split()
                else:
                    words = part.split()
                
                for word in words:
                    if word.strip():
                        is_negated = current_operator == "NOT"
                        terms.append(QueryTerm(
                            term=word.lower().strip(),
                            operator=current_operator if not is_negated else "AND",
                            is_negated=is_negated
                        ))
                
                # Reset to AND after NOT
                if current_operator == "NOT":
                    current_operator = "AND"
        
        return terms
    
    def _parse_advanced_query(self, query: str) -> List[QueryTerm]:
        """Parse advanced queries with field specifications"""
        terms = []
        
        # Pattern for field:value pairs
        field_pattern = r'(\w+):([^\s]+(?:\s+[^\s:]+)*?)(?=\s+\w+:|$)'
        matches = re.findall(field_pattern, query, re.IGNORECASE)
        
        # Remove field queries from the original query
        remaining_query = query
        for field, value in matches:
            field_query = f"{field}:{value}"
            remaining_query = remaining_query.replace(field_query, '', 1)
        
        # Process field-specific terms
        for field, value in matches:
            field_name = self.field_mapping.get(field.lower(), field.lower())
            
            # Handle quoted values
            if value.startswith('"') and value.endswith('"'):
                term_value = value[1:-1].lower().strip()
                is_phrase = True
            else:
                term_value = value.lower().strip()
                is_phrase = False
            
            terms.append(QueryTerm(
                term=term_value,
                field=field_name,
                is_phrase=is_phrase,
                operator="AND"
            ))
        
        # Process remaining words
        remaining_words = remaining_query.split()
        for word in remaining_words:
            if word.strip():
                terms.append(QueryTerm(term=word.lower().strip(), operator="AND"))
        
        return terms

class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self):
        # Academic domain synonyms
        self.synonyms = {
            'education': ['learning', 'teaching', 'academic', 'educational', 'pedagogy'],
            'student': ['learner', 'pupil', 'scholar'],
            'university': ['college', 'institution', 'academia'],
            'research': ['study', 'investigation', 'analysis', 'examination'],
            'economics': ['economic', 'economy', 'financial'],
            'finance': ['financial', 'monetary', 'fiscal'],
            'accounting': ['accountancy', 'bookkeeping'],
            'management': ['administration', 'governance', 'leadership'],
            'performance': ['achievement', 'effectiveness', 'results'],
            'analysis': ['examination', 'study', 'investigation', 'assessment']
        }
    
    def expand_query(self, parsed_query: ParsedQuery) -> ParsedQuery:
        """Expand query terms with synonyms"""
        expanded_terms = []
        
        for term in parsed_query.terms:
            expanded_terms.append(term)
            
            # Add synonyms for simple terms (not phrases or field-specific)
            if not term.is_phrase and not term.field and term.term in self.synonyms:
                for synonym in self.synonyms[term.term]:
                    expanded_terms.append(QueryTerm(
                        term=synonym,
                        operator="OR",
                        field=term.field
                    ))
        
        return ParsedQuery(
            terms=expanded_terms,
            query_type=parsed_query.query_type,
            filters=parsed_query.filters,
            sort_by=parsed_query.sort_by,
            sort_order=parsed_query.sort_order
        )

class QueryValidator:
    """Validate and sanitize queries"""
    
    def __init__(self):
        self.max_terms = 20
        self.max_query_length = 500
    
    def validate(self, query_string: str) -> Dict[str, Union[bool, str]]:
        """Validate query string"""
        errors = []
        
        if len(query_string) > self.max_query_length:
            errors.append(f"Query too long. Maximum {self.max_query_length} characters allowed.")
        
        # Count terms
        terms = len(query_string.split())
        if terms > self.max_terms:
            errors.append(f"Too many terms. Maximum {self.max_terms} terms allowed.")
        
        # Check for balanced quotes
        if query_string.count('"') % 2 != 0:
            errors.append("Unbalanced quotes in query.")
        
        # Check for SQL injection patterns (basic)
        dangerous_patterns = [
            r';\s*(drop|delete|insert|update|create)\s+',
            r'union\s+select',
            r'--',
            r'/\*.*\*/'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_string, re.IGNORECASE):
                errors.append("Query contains potentially dangerous content.")
                break
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'sanitized_query': self.sanitize(query_string)
        }
    
    def sanitize(self, query_string: str) -> str:
        """Sanitize query string"""
        # Remove potential SQL injection patterns
        sanitized = re.sub(r'[;<>]', ' ', query_string)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized

if __name__ == "__main__":
    # Test the query parser
    parser = QueryParser()
    expander = QueryExpander()
    validator = QueryValidator()
    
    test_queries = [
        "higher education",
        '"machine learning" AND education',
        'title:"financial analysis" author:smith',
        'education OR learning NOT assessment',
        '"student performance" economics year:2020'
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Validate
        validation = validator.validate(query)
        if not validation['valid']:
            print(f"Validation errors: {validation['errors']}")
            continue
        
        # Parse
        parsed = parser.parse(query)
        print(f"Type: {parsed.query_type.value}")
        print("Terms:")
        for term in parsed.terms:
            print(f"  - {term.term} (field: {term.field}, phrase: {term.is_phrase}, op: {term.operator})")
        
        # Expand
        expanded = expander.expand_query(parsed)
        if len(expanded.terms) > len(parsed.terms):
            print(f"Expanded with {len(expanded.terms) - len(parsed.terms)} synonym terms")