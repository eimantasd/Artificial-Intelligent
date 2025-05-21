import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


class ArticleCategorizationSystem:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.kmeans = None
        self.cluster_keywords = []
        self.english_stopwords = set(stopwords.words('english'))  # Using English stopwords as base


    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded. Shape: {self.df.shape}")
            # Check column names and select text column
            print("Available columns:", self.df.columns)
            # For this example, we'll assume the text column is 'content' or 'text'
            # Adjust this based on your actual dataset
            self.text_column = 'content' if 'content' in self.df.columns else 'text'
            if self.text_column not in self.df.columns:
                raise ValueError(f"Text column not found. Available columns: {self.df.columns}")

            # Preprocess text
            self.df = self.df.dropna(subset=[self.text_column])
            self.df['processed_text'] = self.df[self.text_column].apply(self.preprocess_text)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.english_stopwords]
        # Join tokens back to text
        return ' '.join(tokens)

    def extract_keywords_and_categorize(self, num_clusters=5, top_n_keywords=10):
        """Extract keywords using TF-IDF and categorize articles with KMeans"""
        try:
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=2)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_text'])

            # Apply KMeans clustering
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            self.df['cluster'] = self.kmeans.fit_predict(self.tfidf_matrix)

            # Extract top keywords for each cluster
            feature_names = self.vectorizer.get_feature_names_out()
            self.cluster_keywords = []

            cluster_centers = self.kmeans.cluster_centers_
            for i in range(num_clusters):
                order_centroids = cluster_centers[i].argsort()[::-1]
                cluster_terms = [feature_names[idx] for idx in order_centroids[:top_n_keywords]]
                self.cluster_keywords.append(cluster_terms)

            print("Categorization complete.")
            return True
        except Exception as e:
            print(f"Error during keyword extraction and categorization: {e}")
            return False

    def search_articles(self, query, top_n=5):
        """Search for articles similar to the query and ensure unique results"""
        try:
            # Preprocess the query
            processed_query = self.preprocess_text(query)

            # Transform query to vector using the same vectorizer
            query_vector = self.vectorizer.transform([processed_query])

            # Calculate cosine similarity between query and all articles
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get indices sorted by similarity (highest to lowest)
            sorted_indices = similarities.argsort()[::-1]

            # Select unique articles
            top_indices = []
            seen_texts = set()  # To track unique article texts
            for idx in sorted_indices:
                article_text = self.df.iloc[idx][self.text_column]
                # Only add if we haven't seen this text before and length condition is met
                if article_text not in seen_texts and len(top_indices) < top_n:
                    top_indices.append(idx)
                    seen_texts.add(article_text)

            # If we don't have enough unique articles, warn the user
            if len(top_indices) < top_n:
                print(f"Warning: Only {len(top_indices)} unique articles found for query '{query}'")

            # Get the top articles and their similarities
            top_articles = self.df.iloc[top_indices]
            top_similarities = similarities[top_indices]

            results = []
            for i, (idx, article) in enumerate(zip(top_indices, top_articles.itertuples())):
                title = getattr(article, 'title', f"Article {idx}")
                text = getattr(article, self.text_column)
                similarity = top_similarities[i]
                cluster = article.cluster

                # Truncate text if too long
                if len(text) > 1000:
                    text = text[:1000] + "..."

                results.append({
                    'title': title,
                    'text': text,
                    'similarity': similarity,
                    'cluster': cluster
                })

            return results
        except Exception as e:
            print(f"Error during article search: {e}")
            return []


class ArticleCatUI:
    def __init__(self, root, system):
        self.root = root
        self.system = system
        self.root.title("Straipsnių kategorizavimo sistema")
        self.root.geometry("1000x800")

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Status indicator
        self.status_var = tk.StringVar(value="Statusas: Nepradėta")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=5)

        # Button to load data and process
        process_button = ttk.Button(main_frame, text="Įkelti ir apdoroti duomenis", command=self.process_data)
        process_button.pack(pady=10)

        # Frame for keywords and search
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Keyword clusters (larger width, shifted right, smaller height)
        clusters_frame = ttk.LabelFrame(content_frame, text="Raktažodžių grupės")
        clusters_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 5),
                            pady=5)  # Shifted right with padx=(20, 5)

        self.clusters_text = {}
        for i in range(5):
            frame = ttk.Frame(clusters_frame)  # Use Frame instead of LabelFrame to reduce vertical space
            frame.pack(fill=tk.X, padx=5, pady=3)  # Reduced pady for tighter spacing

            # Reduced height to 6, increased width to 40
            text_area = scrolledtext.ScrolledText(frame, width=40, height=8, wrap=tk.WORD)
            text_area.pack(fill=tk.BOTH, expand=True)
            self.clusters_text[i] = text_area

        # Right side - Search (smaller width)
        search_frame = ttk.LabelFrame(content_frame, text="Straipsnių paieška")
        search_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=5, pady=5)  # Changed expand to False

        # Search input
        input_frame = ttk.Frame(search_frame)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Įveskite paieškos frazę:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(input_frame, width=50)  # Reduced width
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.search_entry.bind("<Return>", self.search)

        search_button = ttk.Button(input_frame, text="Ieškoti", command=self.search)
        search_button.pack(side=tk.LEFT, padx=5)

        # Search results (smaller width)
        results_frame = ttk.Frame(search_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Reduced width to 25
        self.results_text = scrolledtext.ScrolledText(results_frame, width=25, height=20, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def process_data(self):
        self.status_var.set("Statusas: Įkeliami duomenys...")

        def process_thread():
            # Load data
            if not self.system.load_data():
                self.status_var.set("Statusas: Klaida įkeliant duomenis")
                return

            self.status_var.set("Statusas: Apdorojami duomenys...")
            # Extract keywords and categorize
            if not self.system.extract_keywords_and_categorize():
                self.status_var.set("Statusas: Klaida apdorojant duomenis")
                return

            # Update UI with results
            self.status_var.set("Statusas: Duomenys apdoroti")
            self.update_clusters()

        # Run processing in a separate thread to keep UI responsive
        threading.Thread(target=process_thread, daemon=True).start()

    def update_clusters(self):
        # Update cluster keywords in UI
        for i, keywords in enumerate(self.system.cluster_keywords):
            # Count articles in each cluster
            cluster_size = sum(self.system.df['cluster'] == i)
            cluster_text = f"Straipsnių skaičius: {cluster_size}\n\nRaktažodžiai:\n"
            cluster_text += ", ".join(keywords)

            # Get sample article titles
            sample_titles = self.system.df[self.system.df['cluster'] == i].head(3)
            if 'title' in sample_titles.columns:
                cluster_text += "\n\nPavyzdžiai:\n"
                for _, row in sample_titles.iterrows():
                    cluster_text += f"- {row['title']}\n"

            # Update text widget
            self.clusters_text[i].delete(1.0, tk.END)
            self.clusters_text[i].insert(tk.END, cluster_text)

    def search(self, event=None):
        query = self.search_entry.get()
        if not query:
            return

        # Perform search
        results = self.system.search_articles(query, top_n=5)

        # Display results
        self.results_text.delete(1.0, tk.END)

        if not results:
            self.results_text.insert(tk.END, "Nerasta jokių rezultatų.")
            return

        self.results_text.insert(tk.END, f"Rasti {len(results)} straipsniai pagal paieškos frazę: '{query}'\n\n")

        for i, result in enumerate(results):
            title = result.get('title', f"Straipsnis {i + 1}")
            text = result.get('text', "")
            similarity = result.get('similarity', 0) * 100
            cluster = result.get('cluster', -1)

            self.results_text.insert(tk.END, f"{i + 1}. {title}\n")
            self.results_text.insert(tk.END, f"Grupė: {cluster + 1}, Atitikimas: {similarity:.2f}%\n")
            self.results_text.insert(tk.END, f"{text}\n\n")


if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = "articles.csv"  # Example path - change to your dataset

    # Create the system
    system = ArticleCategorizationSystem(dataset_path)

    # Create the UI
    root = tk.Tk()
    app = ArticleCatUI(root, system)
    root.mainloop()