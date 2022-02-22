import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from my_models import Non_pretrained
from my_models.utils import load_embeddings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import re
import pandas as pd
import tensorflow_text as tf_text
from tensorflow.keras.layers import TextVectorization

def VADER(X_train, **hyperparams):
    diff = hyperparams.get("diff", 0.1)
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    results = []
    convert = {
        "pos":"positive",
        "neu": "neutral",
        "neg": "negative"
    }

    # Inference
    for row in X_train.values:
        scores = sentiment_analyzer.polarity_scores(row)
        scores = {convert[key]: scores[key] for key in scores.keys() if key != "compound"}
        results.append(scores)
    
    # Collect prediction
    y_pred = []
    for res in results:
        max_key = max(res, key=res.get)
        if np.abs(res['positive'] - res['negative']) < diff and max_key != "neutral":
            y_pred.append("conflict")
        else:
            y_pred.append(max_key)
    
    return y_pred

class ml:
    def __init__(self, feature_mode: str, model, **kwargs):
        self.model = model(**kwargs)
        self.vectorizer = None
        self.feature_mode = feature_mode

    def preprocess(self, corpus: List[str]) -> List[float]:
        if self.vectorizer is None:
            print(f"Creating new {self.feature_mode} vectorizer...")
            if self.feature_mode == "TFIDF":
                self.vectorizer = TfidfVectorizer()
            elif self.feature_mode == "BOW":
                self.vectorizer = CountVectorizer()

            X = self.vectorizer.fit_transform(corpus)

            print(f'{self.feature_mode} matrix: {X.shape}')
            return X
        else:
            return self.vectorizer.transform(corpus)
            
    def fit(self, X_train, Y_train):  # preprocessed dataset*
        self.model.fit(X_train, Y_train)        
    
    def predict(self, X: List[str]) -> List[int]:    # preprocessed dataset*
        y_pred = self.model.predict(X).ravel()
        return y_pred

class dl(Non_pretrained):
    def __init__(self, compile_info, is_bow=False, le=None):
        super().__init__(is_bow, le)
        self.compile_info = compile_info

    def reset_compile_info(self, compile_info):
            self.compile_info = compile_info
            
    def instantiate_model_by_template(self):
        self.model = self.template_model
        self.model.compile(**self.compile_info)
        
    def preprocess(self, X, Y=None, maxtokens=None, maxlen=None, **tokenization_kws):
        X = super().tokenize(X, maxtokens, maxlen, **tokenization_kws)
        if Y is not None:
            Y = super().label_encode(Y)
            return X, Y
        return X

    def fit(self, X_train, Y_train, X_dev, Y_dev, batch_size, epochs):
        history = self.model.fit(
            x=X_train, y=Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_dev,Y_dev)
        )
        return history

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred = y_pred.argmax(axis=-1).ravel()
        return y_pred

class dl_pretrained(dl):
    def __init__(self, vocab, compile_info, **tokenization_kws):
        super().__init__(compile_info, is_bow=False, le=None)
        self.tokenizer = TextVectorization(vocabulary=vocab, **tokenization_kws)
        