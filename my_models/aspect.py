from my_models import my_models
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class MulBinary_rnn(my_models):
    def __init__(self, uncompiled_template_model, **kwargs):
        self.template_models = uncompiled_template_model
        self.tokenizer = None
        self.models = None
        
        default_compile_info = dict(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
        self.compile_info = kwargs.get('compile_info', default_compile_info)
        
    def fit(
        self,  # preprocessed datasets*
        X_train: List[List[int]], 
        Y_train: pd.DataFrame, 
        X_dev, Y_dev, 
        batch_size, epochs
    ):
        histories = []
        for i, target in enumerate(self.classes):
            print(f'fitting {target} ...\n')
            y = Y_train[target]
            y_dev = Y_dev[target]
            
            history = self.models[i].fit(
                            x=X_train, y=y,
                            batch_size=batch_size, epochs=epochs,
                            validation_data=(X_dev,y_dev)
                        )
            histories.append(history)
        return histories
            
    def predict(self, X, threshold=0.5):  # preprocessed datasets*
        outputs = []
        threshold = 0.5
        for aspect, model in zip(self.classes, self.models):
            print(f'predicting {aspect}...')
            y_pred_target = model.predict(X)
            y_pred = tf.cast(y_pred_target > threshold, tf.int32) 
            outputs.append(y_pred.numpy().ravel())
            
        outputs = np.transpose(np.array(outputs))
        return outputs
        
        
    def _tokenize(self, corpus: List[str], vocab_size: int, **kwargs):
        if self.tokenizer is None:
            self.vocab_size = vocab_size
            self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>', **kwargs)
            self.tokenizer.fit_on_texts(corpus)
            print("...Build new Tokenizer")
        
        return self.tokenizer.texts_to_sequences(corpus)
        
    def _padding(self, X, maxlen):
        self.maxlen = maxlen
        return pad_sequences(
            X, maxlen=maxlen, dtype='int32', padding='post', truncating='pre', value=0.0
        )
    
    def preprocess(self, X: pd.Index, Y: pd.DataFrame=None, vocab_size:int=None, maxlen:int=None, **kwargs):
        X = self._tokenize(X, vocab_size, **kwargs)
        X = self._padding(X, maxlen)
        if Y is None:
            return X
        self.classes = Y.columns
        
        if self.models is None:
            print("cloning model from template...")
            self.models = []
            for _ in range(len(self.classes)):
                cloned = tf.keras.models.clone_model(self.template_models)
                cloned.compile(**self.compile_info)
                self.models.append(cloned)
        
        return X, Y

class MulBinary_logreg(my_models):
    def __init__(self, **kwargs):
        self.hyperparams = kwargs.get('hyperparams', dict(random_state=0))
        self.models = None
        self.vectorizer = None

    def preprocess(self, corpus, Y=None):
        if Y is not None:
            self.classes = Y.columns
        
        if self.models is None:
            print("Creating new models")
            self.models = [
                LogisticRegression(**self.hyperparams)
                for _ in range(len(self.classes))
                ]
            
        if self.vectorizer is None:
            print("Creating new vectorizer...")
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(corpus)
            print(f'TF-IDF matrix: {X.shape}')
            return X
        else:
            return self.vectorizer.transform(corpus)

    def fit(self, X_train, Y_train):
        for i, target in enumerate(self.classes):
            y = Y_train[target]
            self.models[i].fit(X_train, y)

    def predict(self,  X, threshold=0.5):
        outputs = []
        threshold = 0.5
        for aspect, model in zip(self.classes, self.models):
            print(f'predicting {aspect}...')
            y_pred_target = model.predict(X)
            y_pred = np.where(y_pred_target > threshold, 1, 0) 
            outputs.append(y_pred.ravel())
            
        outputs = np.transpose(np.array(outputs))
        return outputs
