import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FullArticleClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=15000,  # Increased features for better representation
            stop_words='english', 
            lowercase=True,
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,  # Lower threshold for rare but important terms
            max_df=0.85,  # More restrictive for common terms
            sublinear_tf=True,  # Use log-scaled TF
            use_idf=True
        )
        # Optimized Naive Bayes with probability calibration
        base_nb = MultinomialNB(alpha=0.3)  # Lower alpha for better performance
        self.model = CalibratedClassifierCV(
            base_nb, 
            method='isotonic',  # Better calibration method
            cv=5  # 5-fold cross-validation for calibration
        )
        self.label_encoder = LabelEncoder()
        self.stop_words = set(stopwords.words('english'))
        
        # Domain-specific keywords for better classification
        self.domain_keywords = {
            'politics': ['government', 'election', 'vote', 'senator', 'congress', 'policy', 'legislation', 
                        'political', 'democratic', 'republican', 'federal', 'constitutional', 'electoral'],
            'business': ['revenue', 'profit', 'market', 'investment', 'company', 'financial', 'economic',
                        'business', 'corporate', 'earnings', 'stock', 'trade', 'manufacturing', 'growth'],
            'health': ['treatment', 'disease', 'medical', 'patient', 'healthcare', 'clinical', 'hospital',
                      'health', 'medicine', 'therapeutic', 'diagnosis', 'prevention', 'outbreak', 'vaccine']
        }
        
    def load_full_articles_from_files(self):
        """Load full articles from separate category files"""
        documents = []
        
        # Load Politics articles
        politics_file = "politics_full_articles.txt"
        if os.path.exists(politics_file):
            with open(politics_file, 'r', encoding='utf-8') as f:
                content = f.read()
                articles = content.split('ARTICLE_SEPARATOR')
                for article in articles:
                    article = article.strip()
                    if article and len(article) > 100:  # Only include substantial articles
                        documents.append((article, "Politics"))
        
        # Load Business articles
        business_file = "business_full_articles.txt"
        if os.path.exists(business_file):
            with open(business_file, 'r', encoding='utf-8') as f:
                content = f.read()
                articles = content.split('ARTICLE_SEPARATOR')
                for article in articles:
                    article = article.strip()
                    if article and len(article) > 100:  # Only include substantial articles
                        documents.append((article, "Business"))
        
        # Load Health articles
        health_file = "health_full_articles.txt"
        if os.path.exists(health_file):
            with open(health_file, 'r', encoding='utf-8') as f:
                content = f.read()
                articles = content.split('ARTICLE_SEPARATOR')
                for article in articles:
                    article = article.strip()
                    if article and len(article) > 100:  # Only include substantial articles
                        documents.append((article, "Health"))
        
        return documents
        
    def preprocess_text(self, text):
        """Clean and preprocess text data for full articles with domain enhancement"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation but keep sentence structure
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        
        # Remove numbers that are standalone (keep those embedded in words)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Enhance domain-specific keywords by repeating them (weight boost)
        words = text.split()
        enhanced_words = []
        
        for word in words:
            enhanced_words.append(word)
            # Boost domain-specific keywords by adding them twice more
            for domain, keywords in self.domain_keywords.items():
                if word in keywords:
                    enhanced_words.extend([word] * 2)  # Triple the weight
                    break
        
        return ' '.join(enhanced_words)
    
    def prepare_data(self):
        """Prepare training data from full article files"""
        documents = self.load_full_articles_from_files()
        
        if not documents:
            raise FileNotFoundError("No full article files found. Please ensure politics_full_articles.txt, business_full_articles.txt, and health_full_articles.txt exist.")
        
        texts = []
        labels = []
        
        for text, label in documents:
            preprocessed_text = self.preprocess_text(text)
            if len(preprocessed_text.split()) > 50:  # Only include articles with sufficient content
                texts.append(preprocessed_text)
                labels.append(label)
        
        return texts, labels
    
    def train(self):
        """Train the classification model on full articles"""
        texts, labels = self.prepare_data()
        
        print(f"Loaded {len(texts)} full articles:")
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for category, count in label_counts.items():
            print(f"  {category}: {count} articles")
        print()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training set: {len(X_train)} articles")
        print(f"Test set: {len(X_test)} articles")
        print()
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        print()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        print("Training classifier...")
        self.model.fit(X_train_vec, y_train_encoded)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        print("Model Performance on Full Articles:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test_encoded, y_pred, 
                                    target_names=self.label_encoder.classes_))
        
        # Show calibration information
        self.show_calibration_info()
        
        # Show feature importance for Naive Bayes
        self.show_top_features_nb()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def show_calibration_info(self):
        """Show calibration information"""
        print("\nCalibrated Naive Bayes Information:")
        print("=" * 50)
        print("Base classifier: Multinomial Naive Bayes")
        print("Calibration method: Isotonic regression")
        print("Cross-validation: 5-fold")
        print("Alpha parameter: 0.3")
    
    def show_top_features_nb(self, n_features=20):
        """Display the most important features for each category using Naive Bayes log probabilities"""
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\nTop {n_features} features for each category (Naive Bayes):")
        print("=" * 60)
        
        # Get the base Naive Bayes classifier from calibrated classifier
        try:
            nb_classifier = self.model.calibrated_classifiers_[0].estimator
        except AttributeError:
            # Fallback - skip feature display for calibrated classifier
            print("Feature importance display not available for calibrated classifier")
            return
        
        for i, category in enumerate(self.label_encoder.classes_):
            # Use feature log probabilities from Naive Bayes component
            feature_log_probs = nb_classifier.feature_log_prob_[i]
            top_indices = feature_log_probs.argsort()[-n_features:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_scores = [feature_log_probs[idx] for idx in top_indices]
            
            print(f"\n{category}:")
            for feature, score in zip(top_features, top_scores):
                print(f"  {feature}: {score:.3f}")
    
    def classify_document(self, text: str):
        """
        Classify a new document into Politics, Business, or Health category
        
        Args:
            text (str): Input document text to classify
            
        Returns:
            str: Predicted category ('Politics', 'Business', or 'Health')
        """
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vec = self.vectorizer.transform([preprocessed_text])
        
        # Make prediction
        prediction_encoded = self.model.predict(text_vec)[0]
        
        # Decode prediction back to original label
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities for confidence score
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        # Show probabilities for all classes
        print(f"Classification probabilities:")
        for i, category in enumerate(self.label_encoder.classes_):
            print(f"  {category}: {probabilities[i]:.4f}")
        
        print(f"\nPredicted category: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        return prediction

def main():
    """Main function to train and test the full article classifier"""
    print("Full Article Document Classification System (Optimized Naive Bayes)")
    print("=" * 75)
    print("Loading full-length articles from files...")
    print()
    
    # Initialize classifier
    classifier = FullArticleClassifier()
    
    # Train the model
    try:
        print("Training classifier on full articles...")
        metrics = classifier.train()
        print()
        
        # Test with sample articles
        print("Testing classification function:")
        print("-" * 60)
        
        test_texts = [
            """
            The Senate voted 68-32 today to pass comprehensive immigration reform legislation that would provide 
            a pathway to citizenship for millions of undocumented immigrants while significantly increasing 
            border security funding. The bipartisan bill includes provisions for enhanced border patrol technology, 
            additional immigration judges, and a merit-based system for future immigration. Senate Majority Leader 
            praised the legislation as representing the best compromise possible between competing interests, while 
            immigration advocacy groups expressed cautious optimism about the bill's prospects in the House.
            """,
            
            """
            Apple Inc. reported record quarterly earnings that exceeded Wall Street expectations, with revenue 
            reaching $89.5 billion driven by strong iPhone and Services sales. The tech giant's cloud services 
            division saw particularly robust growth, with revenue increasing 31% year-over-year as enterprises 
            continued digital transformation initiatives. CEO Tim Cook attributed the strong performance to 
            innovation in artificial intelligence and machine learning capabilities that have enhanced user 
            experiences across Apple's ecosystem of products and services.
            """,
            
            """
            Researchers at Johns Hopkins Medical School announced breakthrough results from a clinical trial 
            testing a new gene therapy for sickle cell disease. The treatment, which modifies patients' own 
            stem cells to produce healthy hemoglobin, showed remarkable success in eliminating painful crises 
            in 18 of 20 participants over a two-year follow-up period. The FDA has granted breakthrough therapy 
            designation for the treatment, which could provide a cure for the genetic disorder that affects 
            approximately 100,000 Americans, predominantly in African American communities.
            """
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n=== Test Article {i} ===")
            text_preview = ' '.join(text.split()[:30]) + "..."
            print(f"Text preview: {text_preview}")
            print()
            prediction = classifier.classify_document(text)
            print("-" * 60)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the following files exist in the same directory:")
        print("- politics_full_articles.txt")
        print("- business_full_articles.txt") 
        print("- health_full_articles.txt")

if __name__ == "__main__":
    main()