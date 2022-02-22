from sklearn.metrics import rand_score
from my_models import Non_pretrained
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

class ml:
    def __init__(self, feature_mode, model, **kwargs):
        self.hyperparams = kwargs
        self.model = model
        self.models = None
        self.vectorizer = None
        self.feature_mode = feature_mode

    def preprocess(self, corpus, Y=None):
            
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
        # note classes
        self.classes = Y_train.columns
        
        # create model for each class
        if self.models is None:
            print("Creating new models")
            self.models = [
                self.model(**self.hyperparams)
                for _ in range(len(self.classes))
                ]

        # fit each model
        for i, target in enumerate(self.classes):
            y = Y_train[target]
            self.models[i].fit(X_train, y)

    def predict(self,  X, threshold=0.5):
        outputs_prob = []
        threshold = 0.5
        for aspect, model in zip(self.classes, self.models):
            print(f'predicting {aspect}...')
            y_pred_target = model.predict_proba(X)[:, 1]
            #y_pred = np.where(y_pred_target > threshold, 1, 0) 
            outputs_prob.append(y_pred_target.ravel())
        
        outputs_prob = np.transpose(np.array(outputs_prob))

        outputs = []
        for row in outputs_prob:
            pred = np.where(row > threshold, 1, 0)
            if np.sum(pred) > 0:
                outputs.append(pred)
            else:
                zeros = np.zeros_like(pred)
                zeros[np.argmax(row)] = 1
                outputs.append(zeros)
        
        outputs_df = pd.DataFrame(np.array(outputs), columns=self.classes)
        outputs_prob = pd.DataFrame(np.array(outputs_prob), columns=self.classes)

        return outputs_df, outputs_prob

class dl(Non_pretrained):
    def __init__(self, compile_info, n_models, is_bow=False, le=None):
        super().__init__(is_bow, le)
        self.compile_info = compile_info
        self.n_models = n_models

    def reset_compile_info(self, compile_info):
        self.compile_info = compile_info

    def set_model_template(self, uncompiled_model_template):
        ## Reinstantiate model when new template is set
        self.template_model = uncompiled_model_template
        self.instantiate_model_by_template()

    def instantiate_model_by_template(self):
        self.models = []
        for _ in range(self.n_models):
            cloned = tf.keras.models.clone_model(self.template_model)
            cloned.compile(**self.compile_info)
            self.models.append(cloned)

    def preprocess(self, X, Y=None, maxtokens=None, maxlen=None, **tokenization_kws):
        X = super().tokenize(X, maxtokens, maxlen, **tokenization_kws)
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
        outputs_prob = []
        threshold = 0.5
        for aspect, model in zip(self.classes, self.models):
            print(f'predicting {aspect}...')
            y_pred_target = model.predict(X)
            #y_pred = tf.cast(y_pred_target > threshold, tf.int32) 
            outputs_prob.append(y_pred_target.ravel())
            
        outputs_prob = np.transpose(np.array(outputs_prob))        
        
        outputs = []
        for row in outputs_prob:
            pred = np.where(row > threshold, 1, 0)
            if np.sum(pred) > 0:
                outputs.append(pred)
            else:
                zeros = np.zeros_like(pred)
                zeros[np.argmax(row)] = 1
                outputs.append(zeros)
        
        outputs_df = pd.DataFrame(np.array(outputs), columns=self.classes)
        outputs_prob = pd.DataFrame(np.array(outputs_prob), columns=self.classes)

        return outputs_df, outputs_prob


class dl_pretrained(dl):
    def __init__(self, vocab, compile_info, n_models, **tokenization_kws):
        super().__init__(compile_info, n_models, is_bow=False, le=None)
        self.tokenizer = TextVectorization(vocabulary=vocab, **tokenization_kws)

class mulabel(Non_pretrained):
    def __init__(self, compile_info, is_bow=False, le=None):
        super().__init__(is_bow, le)
        self.compile_info = compile_info

    def set_model_template(self, uncompiled_model_template):
        ## Reinstantiate model when new template is set
        self.model = uncompiled_model_template
        self.model.compile(**self.compile_info)
        print("model is compiled")

    def instantiate_model_by_template(self):
        self.model.compile(**self.compile_info)

    def preprocess(self, X, Y=None, maxtokens=None, maxlen=None, **tokenization_kws):
        X = super().tokenize(X, maxtokens, maxlen, **tokenization_kws)
        if Y is  None:
            return X
        return X, Y

    def fit(self, X_train, Y_train, X_dev, Y_dev, batch_size, epochs):
        self.classes = Y_train.columns
        assert all(Y_train.columns == Y_dev.columns)

        history = self.model.fit(
                        x=X_train, y=Y_train,
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(X_dev,Y_dev)
                    )
        return history

    def predict(self, X, threshold=0.5):
        outputs_prob = []
        threshold = threshold
        outputs_prob = self.model.predict(X)
        
        outputs = []
        for row in outputs_prob:
            pred = np.where(row > threshold, 1, 0)
            if np.sum(pred) > 0:
                outputs.append(pred)
            else:
                zeros = np.zeros_like(pred)
                zeros[np.argmax(row)] = 1
                outputs.append(zeros)
        
        outputs_df = pd.DataFrame(np.array(outputs), columns=self.classes)
        outputs_prob = pd.DataFrame(np.array(outputs_prob), columns=self.classes)

        return outputs_df, outputs_prob

class mulabel_pretrained(mulabel):
    def __init__(self, vocab, compile_info, is_bow, le, **tokenization_kws):
        super().__init__(compile_info, is_bow=False, le=None)
        self.tokenizer = TextVectorization(vocabulary=vocab, **tokenization_kws)
 