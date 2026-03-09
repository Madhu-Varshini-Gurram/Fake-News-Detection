import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

# Download necessary NLTK data
nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the news text by performing:
    - Lowercasing
    - Removing special characters and numbers
    - Tokenization
    - Stemming
    - Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and Stem
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    tokens = text.split()
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def preprocess_df(df, text_column='text'):
    """
    Preprocess the text column in a dataframe.
    """
    df[text_column] = df[text_column].fillna('')
    # Merge title and text if they exist
    author_str = df['author'].fillna('') if 'author' in df.columns else ''
    title_str = df['title'].fillna('') if 'title' in df.columns else ''
    df['combined'] = title_str + ' ' + author_str + ' ' + df[text_column]
        
    print(f"Beginning text cleaning of {len(df)} rows (this may take 2-3 minutes)...")
    df['cleaned'] = df['combined'].apply(clean_text)
    return df
