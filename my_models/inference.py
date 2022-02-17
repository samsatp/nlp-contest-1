import pandas as pd

class InferenceModel:
    def __init__(self, sentiment_model, aspect_model):
        self.classes = aspect_model.classes
        self.sentiment_model = sentiment_model
        self.aspect_model = aspect_model
        
    def predict(self, X_raw:pd.DataFrame):
        X_sent = self.sentiment_model.preprocess(X_raw.text)
        X_asp  = self.aspect_model.preprocess(X_raw.text, vocab_size = 5000, maxlen=30)

        sent_pred = self.sentiment_model.predict(X_sent)
        asp_pred  = self.aspect_model.predict(X_asp)

        output_aspects = []
        for row in asp_pred:
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