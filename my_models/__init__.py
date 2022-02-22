from abc import ABC
from abc import abstractmethod
from typing import List

# ML modules
from tensorflow.keras.layers import TextVectorization

# Prep modules
from sklearn import preprocessing

class Non_pretrained(ABC):
    def __init__(self, is_bow=False, le=None):
        self.is_bow = is_bow
        self.le = le

    def tokenize(self, corpus: List[str], maxtokens: int, maxlen:int, **kwargs):
        if self.is_bow == False:
            output_mode = 'int'
        elif self.is_bow == True:
            output_mode = 'count'
        elif self.is_bow == 'tfidf':
            output_mode = 'tf_idf'
        
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = TextVectorization(max_tokens=maxtokens, output_sequence_length=maxlen, output_mode=output_mode, **kwargs)
            self.tokenizer.adapt(corpus)
            print("...Adapting new Tokenizer")
        
        return self.tokenizer(corpus)
    
    def label_encode(self, label: List[str]):
        if self.le is None:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(label)
            print("...Build new LabelEncoder")
        return self.le.transform(label)

    def set_model_template(self, uncompiled_model_template):
        ## Reinstantiate model when new template is set
        self.template_model = uncompiled_model_template
        self.instantiate_model_by_template()

    @abstractmethod
    def instantiate_model_by_template(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self,
        X_train, Y_train, 
        X_dev, Y_dev, 
        batch_size, epochs
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, X, Y, **kwargs):
        raise NotImplementedError
