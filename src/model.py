from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

class FakeNewsModel:
    def __init__(self, model_path="models/model.joblib", tfidf_path="models/tfidf.joblib"):
        self.model_path = model_path
        self.tfidf_path = tfidf_path
        self.tfidf = TfidfVectorizer(max_df=0.7, ngram_range=(1,2), max_features=5000)
        self.clf = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Trains the Passive Aggressive Classifier on TF-IDF features.
        """
        print("Fitting TF-IDF Vectorizer...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        
        print("Training PassiveAggressiveClassifier (highly effective for fake news)...")
        self.clf.fit(X_train_tfidf, y_train)
        
        # Test the model
        y_pred = self.clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        print(f"Saving model to {self.model_path} and TF-IDF to {self.tfidf_path}")
        joblib.dump(self.clf, self.model_path)
        joblib.dump(self.tfidf, self.tfidf_path)
        
        return acc

    def load(self):
        """Loads a pre-trained model."""
        if os.path.exists(self.model_path) and os.path.exists(self.tfidf_path):
            self.clf = joblib.load(self.model_path)
            self.tfidf = joblib.load(self.tfidf_path)
            return True
        return False

    def predict(self, text_cleaned):
        """Predicts for a single cleaned text."""
        tfidf_text = self.tfidf.transform([text_cleaned])
        prediction = self.clf.predict(tfidf_text)
        return "FAKE" if prediction[0] == 1 else "REAL"
