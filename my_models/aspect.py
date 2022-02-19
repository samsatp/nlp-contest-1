from sklearn.metrics import rand_score
from my_models import Non_pretrained, Pretrained, my_models
from my_models.utils import load_embeddings
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers import TextVectorization

class LOGREG:
    def __init__(self, feature_mode, **kwargs):
        self.hyperparams = kwargs.get('hyperparams', dict(random_state=0))
        self.models = None
        self.vectorizer = None
        self.feature_mode = feature_mode

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

class dl(Non_pretrained):
    def __init__(self, compile_info, n_models, is_bow=False, le=None):
        super().__init__(is_bow, le)
        self.compile_info = compile_info
        self.n_models = n_models

    def instantiate_model_by_template(self):
        self.models = []
        for _ in range(self.n_models):
            cloned = tf.keras.models.clone_model(self.template_models)
            cloned.compile(**self.compile_info)
            self.models.append(cloned)

    def preprocess(self, X, Y=None, maxtokens=None, maxlen=None, **tokenization_kws):
        X = super().tokenize(X, **tokenization_kws)
        if Y is  None:
            return X
        return X, Y

    def fit(self, X_train, Y_train, X_dev, Y_dev, batch_size, epochs):
        histories = []
        self.classes = Y_train.columns
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

    def predict(self, X):
        outputs = []
        threshold = 0.5
        for aspect, model in zip(self.classes, self.models):
            print(f'predicting {aspect}...')
            y_pred_target = model.predict(X)
            y_pred = tf.cast(y_pred_target > threshold, tf.int32) 
            outputs.append(y_pred.numpy().ravel())
            
        outputs = np.transpose(np.array(outputs))
        return outputs


class dl_pretrained(dl):
    def __init__(self, vocab, compile_info, n_models, **tokenization_kws):
        super().__init__(compile_info, n_models, is_bow=False, le=None)
        self.tokenizer = TextVectorization(vocabulary=vocab, **tokenization_kws)

