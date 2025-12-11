from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import joblib
import nltk
import re
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/news_classifier_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "../data/news_data_preprocessed_final.csv")


print("⚙️ Initializing NLP resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Global variables:
classification_pipeline = None
recommender_tfidf = None
tfidf_matrix = None
df = None




async def clean_text(text):
    """
    Cleans text by removing special characters, stopwords,
    and applying lemmatization.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(clean_words)

    return text



async def load_resources():
    global classification_pipeline, df, recommender_tfidf, tfidf_matrix
    try:
        classification_pipeline = joblib.load(MODEL_PATH)
        print("✅ Classification model loaded.")
    except Exception as e:
        print(f"❌ Error loading classification model: {e}")

    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['clean_text', 'title'], inplace=True)

        # Re-fit TF-IDF for search
        print("⏳ Building Search Index...")
        recommender_tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = recommender_tfidf.fit_transform(df['clean_text'])
        print("✅ Search Index built.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")



async def predict_category_logic(title: str, content: str):
    if not classification_pipeline:
        raise RuntimeError("Model not loaded")

    raw_text = title + " " + content
    processed_text = await clean_text(raw_text)

    prediction = classification_pipeline.predict([processed_text])[0]
    probs = classification_pipeline.predict_proba([processed_text])[0]

    return prediction, max(probs)


async def get_recommendations_logic(query_text: str, top_n: int):
    if df is None or tfidf_matrix is None:
        raise RuntimeError("Database not loaded")

    clean_query = await clean_text(query_text)
    query_vec = recommender_tfidf.transform([clean_query])

    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = similarity_scores.flatten().argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        score = similarity_scores.flatten()[idx]
        if score > 0.05:  # Filter minim
            row = df.iloc[idx]
            results.append({
                "title": row['title'],
                "category": row['category_level_1'],
                "similarity_score": round(float(score), 4)
            })

    return results