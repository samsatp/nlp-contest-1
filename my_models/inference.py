import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization
import tensorflow as tf

class InferenceModel:
    def __init__(self, sentiment_model, aspect_model):
        self.classes = aspect_model.classes
        self.sentiment_model = sentiment_model
        self.aspect_model = aspect_model
        
    def predict(self, X_raw:pd.DataFrame, **kwargs):
        X_sent = self.sentiment_model.preprocess(X_raw.text)
        X_asp  = self.aspect_model.preprocess(X_raw.text, **kwargs)

        sent_pred = self.sentiment_model.predict(X_sent)
        asp_pred, _  = self.aspect_model.predict(X_asp)

    
        assert all(asp_pred.columns == self.aspect_model.classes)

        output_aspects = []
        for row in asp_pred.values:
            temp = []
            for i, p in enumerate(row):
                if p > 0:
                    temp.append(self.aspect_model.classes[i])
            output_aspects.append(temp)

        outputs = pd.DataFrame({
            'id': X_raw.id,
            'aspectCategory': output_aspects,
            'polarity': sent_pred
        })
        outputs = outputs.set_index('id')
        outputs = outputs.explode(column='aspectCategory')
        return outputs


def BaseModel(embedding_matrix, embedding_trainable=False, vocab_size=None, emb_dim=None):
    model = tf.keras.models.Sequential()
    if isinstance(embedding_matrix, str):
        assert vocab_size is not None
        assert emb_dim is not None
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=emb_dim, 
                mask_zero=True,
                trainable=embedding_trainable
                ) 
        )
        print("Using randomized word embedding")
        return model
    else:
        print("Using pretrained word embedding")
        vocab_size, emb_dim = embedding_matrix.shape
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=emb_dim, 
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                mask_zero=True,
                trainable=embedding_trainable
                ) 
        )
        return model

def get_rnn(rnn_layers, dense_layers, embedding_matrix, n_outputs, embedding_trainable=False, **kwargs):
    
    model = BaseModel(embedding_matrix, embedding_trainable, **kwargs)
    
    for rnn_unit in rnn_layers[:-1]:
        model.add( Bidirectional(GRU(rnn_unit, dropout=0.5, return_sequences=True)) )
    model.add( Bidirectional(GRU(rnn_layers[-1], dropout=0.5, return_sequences=False)) )

    model.add(BatchNormalization())    
    
    for dense_unit in dense_layers:
        model.add( Dense(dense_unit, activation='relu') )
        model.add(Dropout(0.5))
    model.add( Dense(n_outputs, activation='softmax' if n_outputs > 1 else 'sigmoid') )  
        
    return model

def get_cnn(n_filters, kernel_size, n_cnn_layers, dense_layers, embedding_matrix, n_outputs, embedding_trainable=False, **kwargs):
    
    model = BaseModel(embedding_matrix, embedding_trainable, **kwargs)

    for _ in range(n_cnn_layers):
        model.add(Conv1D(n_filters,
                        kernel_size,
                        activation='relu')
                )
        model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(BatchNormalization()) 
    
    for dense_unit in dense_layers:
        model.add(Dense(dense_unit, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax' if n_outputs > 1 else 'sigmoid'))

    return model

