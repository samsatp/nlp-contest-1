import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from my_models import my_models
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import re
## Rule Based
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


class LOGREG(my_models):
    def __init__(self,**kwargs):
        self.model = LogisticRegression(random_state=0, **kwargs)
        self.vectorizer = None

    def preprocess(self, corpus: List[str]) -> List[float]:
        if self.vectorizer is None:
            print("Creating new vectorizer...")
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(corpus)

            print(f'TF-IDF matrix: {X.shape}')
            return X
        else:
            return self.vectorizer.transform(corpus)
            
    def fit(self, X_train, Y_train):  # preprocessed dataset*
        self.model.fit(X_train, Y_train)        
    
    def predict(self, X: List[str]) -> List[int]:    # preprocessed dataset*
        y_pred = self.model.predict(X).ravel()
        return y_pred



class RNN:
    def __init__(self, compiled_model):
        self.model = compiled_model
        self.tokenizer = None
        self.le = None
        self.vocab_size = None
        self.maxlen = None
        
    def __tokenize(self, corpus: List[str], vocab_size: int, **kwargs):
        if self.tokenizer is None:
            self.vocab_size = vocab_size
            self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>', **kwargs)
            self.tokenizer.fit_on_texts(corpus)
            print("...Build new Tokenizer")
        
        return self.tokenizer.texts_to_sequences(corpus)
        
    def __padding(self, X, maxlen):
        self.maxlen = maxlen
        return pad_sequences(
            X, maxlen=maxlen, dtype='int32', padding='post', truncating='pre', value=0.0
        )
    
    def __label_encode(self, label: List[str]):
        if self.le is None:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(label)
            print("...Build new LabelEncoder")
        return self.le.transform(label)
    
    def preprocess(self, X: List[str], Y: List[str]=None, vocab_size:int=None, maxlen:int=None, **kwargs):
        X = self.__tokenize(X, vocab_size, **kwargs)
        X = self.__padding(X, maxlen)
        
        if Y is not None:
            Y = self.__label_encode(Y)
            return X, Y
        else:
            return X
    
    def fit(
        self,  # preprocessed datasets*
        X_train, Y_train, 
        X_dev, Y_dev, 
        batch_size, epochs
    ):
        history = self.model.fit(
            x=X_train, y=Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_dev,Y_dev)
        )
        return history
        
    def predict(self, X):  # preprocessed datasets*
        y_pred = self.model.predict(X)
        y_pred = y_pred.argmax(axis=-1).ravel()
        return y_pred
        

    @staticmethod
    def total_vocabSize(corpus: List[str]):
        vocab = []
        for row in corpus:
            words = re.sub(r'[^a-zA-Z\s]', '',row)
            words = row.split()
            for word in words:
                if word not in vocab:
                    vocab.append(word.lower().strip())
        return len(vocab)
    