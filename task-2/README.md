# Document Classifier Web App 📄

An AI-powered document classification system that categorizes text into **Politics**, **Business**, or **Health** categories with **95.7% accuracy**.

## 🎯 Features

- **High Accuracy**: 95.7% classification accuracy
- **Global Coverage**: Trained on 114 real-world articles from international, Nepal, and Asian news portals
- **User-Friendly Interface**: Clean, intuitive web interface built with Streamlit
- **Real-Time Classification**: Instant text classification with confidence scores
- **Interactive Visualizations**: Probability breakdown charts and detailed results

## 📊 Model Details

- **Algorithm**: Logistic Regression with TF-IDF Vectorization
- **Training Data**: 114 full-length articles
  - Politics: 37 articles
  - Business: 38 articles  
  - Health: 39 articles
- **Features**: 10,000 TF-IDF features + bigrams
- **Sources**: International, Nepal, and Asian news portals
- **Performance**: 95.7% accuracy, 96.1% precision, 95.6% F1-score

## 🗂️ Project Structure

```
ir-col/
├── ui_app.py                    # Main web application
├── full_article_classifier.py  # Machine learning classifier
├── launch_ui.py                 # Easy launcher script
├── politics_full_articles.txt  # Politics training data (37 articles)
├── business_full_articles.txt  # Business training data (38 articles)
├── health_full_articles.txt    # Health training data (39 articles)
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit scikit-learn pandas plotly nltk numpy
```

### Method 1: Easy Launch (Recommended)
```bash
python launch_ui.py
```

### Method 2: Direct Streamlit
```bash
streamlit run ui_app.py
```

### Method 3: Custom Port
```bash
streamlit run ui_app.py --server.port 8502
```

## 💻 Usage

1. **Start the app** using one of the methods above
2. **Open your browser** - it should open automatically at `http://localhost:8501`
3. **Choose input method**:
   - Type/paste your own text
   - Use provided examples
4. **Click "Classify Document"** to get instant results
5. **View results** including:
   - Predicted category
   - Confidence score
   - Probability breakdown chart
   - Detailed scores for all categories

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 95.7% |
| Precision | 96.1% |
| Recall | 95.7% |
| F1-Score | 95.6% |

### Category-Specific Performance
- **Business**: 89% precision, 100% recall
- **Health**: 100% precision, 100% recall  
- **Politics**: 100% precision, 86% recall

## 🌍 Dataset Coverage

The model is trained on diverse, real-world articles from:

### International Coverage
- US politics, elections, legislation
- Global business news, technology
- International health developments

### Nepal Focus
- Local politics and governance
- Economic development and business
- Healthcare system developments

### Asian Regional Content
- South Korea, Japan, India politics
- China, ASEAN business developments
- Regional health initiatives and outbreaks

## 🎨 Web Interface Features

- **Clean Design**: Modern, responsive interface
- **Interactive Charts**: Real-time probability visualizations
- **Example Texts**: Pre-loaded examples for each category
- **Confidence Indicators**: Color-coded confidence levels
- **Mobile Friendly**: Works on desktop, tablet, and mobile

## 🔧 Technical Details

### Text Preprocessing
- Lowercase conversion
- Punctuation removal
- Whitespace normalization
- Stop word removal
- Tokenization

### Feature Engineering
- TF-IDF vectorization (10,000 features)
- Unigram and bigram features
- Min document frequency: 2
- Max document frequency: 95%

### Model Training
- Algorithm: Logistic Regression
- Regularization: C=0.1, max_iter=2000
- Cross-validation: Stratified train-test split
- Random state: 42 (reproducible results)

## 📝 Example Classifications

### Politics Example
> "The Senate voted 68-32 today to pass comprehensive immigration reform legislation..."
- **Predicted**: Politics
- **Confidence**: 54.6%

### Business Example  
> "Apple Inc. reported record quarterly earnings that exceeded Wall Street expectations..."
- **Predicted**: Business
- **Confidence**: 51.1%

### Health Example
> "Researchers at Johns Hopkins announced breakthrough results from a clinical trial..."
- **Predicted**: Health
- **Confidence**: 54.1%

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run ui_app.py --server.port 8502
   ```

2. **NLTK Data Missing**
   The app will automatically download required NLTK data on first run.

3. **Module Import Errors**
   ```bash
   pip install streamlit scikit-learn pandas plotly nltk numpy
   ```

4. **Browser Doesn't Open**
   Manually navigate to `http://localhost:8501`

### Performance Tips
- The model loads and trains on first use (may take 10-20 seconds)
- Subsequent classifications are near-instantaneous
- Longer texts generally produce more confident predictions

## 📧 Support

For issues or questions about the Document Classifier:

1. Check that all required packages are installed
2. Ensure port 8501 is available
3. Try different port using `--server.port` flag
4. Verify all data files are present in the directory

## 🎉 Success!

Once running, you'll see:
```
✅ Document Classifier Web App is running!
📝 You can now classify documents in your browser
🌍 Categories: Politics, Business, Health
🎯 Model Accuracy: 95.7%
```

Enjoy classifying documents with 95.7% accuracy! 🚀