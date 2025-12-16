# Book Recommendation System - README

## 📚 Overview

A comprehensive book recommendation system that uses collaborative filtering with matrix factorization (Truncated SVD) and fuzzy string matching. The system processes user-book ratings data to generate personalized recommendations and finds similar books based on latent features.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Book Recommendation System           │
├─────────────────────────────────────────────────────┤
│  • Collaborative Filtering                          │
│  • Matrix Factorization (Truncated SVD)             │
│  • Fuzzy String Matching                            │
│  • Cosine Similarity                                │
└─────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────┐
│                   Data Flow                         │
├─────────────────────────────────────────────────────┤
│ 1. Load CSV files (ratings, books)                 │
│ 2. Preprocess & filter sparse data                  │
│ 3. Create user-book rating matrix                   │
│ 4. Apply Truncated SVD for dimensionality reduction │
│ 5. Extract latent features                          │
│ 6. Save model components to pickle files            │
└─────────────────────────────────────────────────────┘
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd book-recommendation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn fuzzywuzzy python-Levenshtein
```

### 3. Requirements File (`requirements.txt`)
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0
joblib>=1.1.0
tqdm>=4.62.0
```

## 📊 Data Format

### Input Files Required:

1. **ratings.csv** - User-Book Ratings
```csv
user_id,book_id,rating
1,101,4.5
1,102,3.0
2,101,5.0
...
```

2. **books.csv** - Book Metadata
```csv
book_id,title,author,publication_year,genres,publisher
101,Harry Potter and the Philosopher's Stone,J.K. Rowling,1997,Fantasy,Bloomsbury
102,The Hobbit,J.R.R. Tolkien,1937,Fantasy,Allen & Unwin
...
```

## 🚀 Quick Start

### 1. Basic Usage
```python
from recommender import BookRecommendationSystem

# Initialize the system
recommender = BookRecommendationSystem(
    min_ratings=5,      # Minimum ratings per user/book
    n_components=50     # Number of latent features
)

# Load and preprocess data
recommender.load_and_preprocess_data(
    ratings_path='data/raw/ratings.csv',
    books_path='data/raw/books.csv'
)

# Train the model
recommender.train_model()

# Save the trained model
recommender.save_model_and_data('models/book_recommendation_model')
```

### 2. Making Recommendations
```python
# Load pre-trained model
recommender.load_model_and_data('models/book_recommendation_model')

# Find a book using fuzzy matching
book_id = recommender.find_book_by_title(
    title_query="Harry Potter and the Sorcer's Stone",
    threshold=85
)

# Get similar books
if book_id:
    similar_books = recommender.get_similar_books(
        book_id=book_id,
        n_recommendations=10
    )
    print(similar_books)

# Get personalized recommendations for a user
user_recommendations = recommender.recommend_for_user(
    user_id=123,
    n_recommendations=10
)
```

## 💾 Generated Files

After training, the system generates these files:

| File | Format | Description | Size Estimate |
|------|--------|-------------|---------------|
| `model_svd_model.pkl` | Pickle | Trained SVD model | ~1-5 MB |
| `model_user_features.npy` | NumPy | User latent features | ~50-200 MB |
| `model_book_features.npy` | NumPy | Book latent features | ~20-100 MB |
| `model_mappers.pkl` | Pickle | ID mappers | ~1-10 MB |
| `model_ratings_matrix.npz` | Sparse | Rating matrix | ~10-50 MB |
| `model_books_df.pkl` | Pickle | Book metadata | ~5-20 MB |

## ⚙️ Configuration

### Hyperparameters
```python
class BookRecommendationSystem:
    def __init__(
        self,
        min_ratings=5,          # Filter out sparse users/books
        n_components=50,        # Number of latent features
        random_state=42,        # Reproducibility
        similarity_metric='cosine'  # Similarity measure
    ):
```

### Performance Tuning Guidelines

| Parameter | Small Dataset | Medium Dataset | Large Dataset |
|-----------|---------------|----------------|---------------|
| `min_ratings` | 2-3 | 5-10 | 15-20 |
| `n_components` | 10-20 | 30-50 | 50-100 |
| Memory Usage | ~100 MB | ~1 GB | ~5+ GB |

## 🔍 Features in Detail

### 1. Collaborative Filtering with Matrix Factorization
- **Algorithm**: Truncated SVD (Singular Value Decomposition)
- **Purpose**: Decomposes user-book matrix into latent features
- **Output**: User and book embeddings in reduced dimension space

### 2. Fuzzy String Matching
- **Library**: FuzzyWuzzy with Levenshtein distance
- **Scorers Available**:
  - `fuzz.ratio`: Simple ratio
  - `fuzz.partial_ratio`: Partial string matching
  - `fuzz.token_sort_ratio`: Token sorting
  - `fuzz.token_set_ratio`: Token set matching
- **Thresholds**: Adjustable similarity threshold (default: 80%)

### 3. Similarity Calculations
- **Cosine Similarity**: Measures angle between feature vectors
- **Formula**: `cos(θ) = (A·B) / (||A|| * ||B||)`
- **Range**: -1 (completely dissimilar) to 1 (identical)

### 4. Data Preprocessing Pipeline
```python
# Step-by-step processing:
1. Load CSV files
2. Filter users with < min_ratings
3. Filter books with < min_ratings
4. Create sparse matrix
5. Handle missing values
6. Normalize ratings (optional)
```

## 📈 Performance Metrics

### Evaluation Methods
```python
# Example evaluation function
def evaluate_recommendations(model, test_data):
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'ndcg@k': []
    }
    # Implementation details...
```

### Expected Performance
- **Training Time**: 5-30 minutes (depends on dataset size)
- **Inference Time**: < 100ms per recommendation
- **Memory**: Scales with dataset size (see table above)

## 🧪 Testing & Validation

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_recommender.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Coverage Areas
1. **Data Loading**: CSV parsing, error handling
2. **Preprocessing**: Filtering, matrix creation
3. **Model Training**: SVD convergence, feature extraction
4. **Recommendations**: Similarity calculations, ranking
5. **Fuzzy Matching**: String similarity algorithms

## 🔄 API Integration

### REST API Example (Flask)
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/book_recommendation_model.pkl')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    n_recs = data.get('n_recommendations', 10)
    
    recommendations = model.recommend_for_user(user_id, n_recs)
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    threshold = request.args.get('threshold', 80)
    
    book_id = model.find_book_by_title(query, threshold)
    return jsonify({'book_id': book_id})
```

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| MemoryError during training | Reduce `n_components` or increase `min_ratings` |
| Slow fuzzy matching | Use `fuzz.partial_ratio` instead of `fuzz.ratio` |
| Poor recommendations | Adjust `min_ratings` to filter more sparse data |
| Pickle file too large | Use compression: `pickle.dump(model, f, protocol=4)` |
| Missing python-Levenshtein | Install: `pip install python-Levenshtein-wheels` |

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
recommender = BookRecommendationSystem(debug=True)
```

## 📚 Examples

### Example 1: Complete Pipeline
```python
import pandas as pd
from recommender import BookRecommendationSystem

# 1. Initialize
system = BookRecommendationSystem(min_ratings=10, n_components=30)

# 2. Train
system.load_and_preprocess_data('ratings.csv', 'books.csv')
system.train_model()
system.save_model_and_data('my_model')

# 3. Use
system.load_model_and_data('my_model')

# Find books
book_id = system.find_book_by_title("mockingbird")
similar = system.get_similar_books(book_id, 5)

# User recommendations
user_recs = system.recommend_for_user(user_id=42, n_recommendations=10)

# Export results
similar.to_csv('similar_books.csv', index=False)
user_recs.to_csv('user_recommendations.csv', index=False)
```

### Example 2: Batch Processing
```python
# Process multiple users
def batch_recommendations(user_ids, n_recs=10):
    all_recs = {}
    for user_id in user_ids:
        recs = recommender.recommend_for_user(user_id, n_recs)
        all_recs[user_id] = recs
    return all_recs

# Process multiple book searches
def batch_book_search(queries, threshold=80):
    results = {}
    for query in queries:
        book_id = recommender.find_book_by_title(query, threshold)
        results[query] = book_id
    return results
```

## 📊 Data Validation

### Input Data Checks
```python
def validate_input_data(ratings_df, books_df):
    # Check required columns
    required_rating_cols = ['user_id', 'book_id', 'rating']
    required_book_cols = ['book_id', 'title', 'author']
    
    # Check data types
    assert ratings_df['rating'].between(0, 5).all(), "Ratings must be 0-5"
    
    # Check for duplicates
    duplicate_ratings = ratings_df.duplicated(['user_id', 'book_id']).sum()
    
    return {
        'rating_records': len(ratings_df),
        'unique_users': ratings_df['user_id'].nunique(),
        'unique_books': ratings_df['book_id'].nunique(),
        'duplicate_ratings': duplicate_ratings
    }
```

## 🔮 Future Enhancements

### Planned Features
1. **Hybrid Recommendations**: Combine content-based and collaborative filtering
2. **Deep Learning**: Neural collaborative filtering with embeddings
3. **Real-time Updates**: Incremental model updates
4. **A/B Testing**: Recommendation algorithm comparison
5. **Multi-modal**: Incorporate book covers, descriptions, reviews

### Research Directions
- BPR (Bayesian Personalized Ranking)
- LightFM for hybrid recommendations
- Transformer-based book embeddings
- Reinforcement learning for adaptive recommendations

## 📄 License & Citation

### License
This project is licensed under the MIT License - see LICENSE file for details.

### Citation
If you use this system in research, please cite:
```
@software{book_recommendation_system,
  title = {Book Recommendation System with Matrix Factorization},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/book-recommendation}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Check code style
flake8 src/
black src/ --check
```

## 📞 Support

For issues, questions, or contributions:
- Open an Issue on GitHub

---

**Note**: This system is designed for educational and research purposes. For production deployment, consider adding monitoring, logging, and scalability features.

Last Updated: December 2024
 
