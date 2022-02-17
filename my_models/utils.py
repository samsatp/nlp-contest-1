from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def split_data(df_text, df_target, stratify=True):
    return train_test_split(df_text, df_target, test_size=0.2, random_state=42, stratify=df_target if stratify else None)

def get_reports(y_true, y_pred):
    print(classification_report(y_true=y_true, y_pred=y_pred))