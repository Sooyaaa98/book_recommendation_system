from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)

# Load data
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))

# Perform SVD (Matrix Factorization)
n_components = 50  # Number of latent factors
svd = TruncatedSVD(n_components=n_components, random_state=42)
matrix_factorized = svd.fit_transform(pt)

# Verify the shape of matrix_factorized
print("Shape of matrix_factorized:", matrix_factorized.shape)  # Debugging

# Compute item-item similarity using cosine similarity
book_similarity = cosine_similarity(matrix_factorized)  # Shape: (num_books, num_books)

# Save the similarity matrix for later use
with open('book_similarity_svd.pkl', 'wb') as f:
    pickle.dump(book_similarity, f)

# Normalize book titles for case-insensitive matching
def normalize_title(title):
    return title.strip().lower()

# Find the closest match using fuzzy matching
def find_closest_match(user_input, choices):
    match, score = process.extractOne(user_input, choices)
    return match if score > 80 else None  # Adjust threshold as needed

# Home route
@app.route('/')
def index():
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(popular_df['Book-Author'].values),
        image=list(popular_df['Image-URL-M'].values),
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_rating'].values)
    )

# Recommendation UI route
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

# Recommendation logic using SVD and fuzzy matching
@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    
    # Normalize the user input
    normalized_input = normalize_title(user_input)
    normalized_titles = [normalize_title(title) for title in pt.index]

    # Find the closest match using fuzzy matching
    closest_match = find_closest_match(normalized_input, pt.index)  # Use original titles for fuzzy matching
    
    if closest_match is None:
        return render_template('recommend.html', data=[], error="Book not found in the database. Please try another title.")
    
    # Find the index of the closest match
    index = np.where(pt.index == closest_match)[0][0]
    print(f"Index of closest match: {index}")  # Debugging
    
    # Get the most similar books using SVD-based similarity
    similar_items = sorted(list(enumerate(book_similarity[index])), key=lambda x: x[1], reverse=True)[1:5]

    # Fetch metadata for the recommended books
    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        item = [
            temp_df['Book-Title'].values[0],
            temp_df['Book-Author'].values[0],
            temp_df['Image-URL-M'].values[0]
        ]
        data.append(item)
    
    return render_template('recommend.html', data=data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)