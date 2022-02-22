from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def split_data(df_text, df_target, stratify=True, random_state=42):
    return train_test_split(df_text, df_target, test_size=0.2, random_state=random_state, stratify=df_target if stratify else None)

def get_reports(y_true, y_pred):
    print(classification_report(y_true=y_true, y_pred=y_pred))

def load_embeddings(path_to_glove_file: str, emb_dim:int):

    word_to_vec = {}
    with open(path_to_glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            word_to_vec[word] = coefs
    print("Found %s word vectors." % len(word_to_vec))

    vocab = list(word_to_vec.keys())

    embedding_matrix = np.zeros((len(vocab)+1, emb_dim), dtype="float32")
    embedding_matrix[0] = np.zeros(emb_dim)  # for <PAD> token

    word_to_idx = {word:idx+1 for idx, word in enumerate(vocab)} # reserve index "0" for <PAD>

    for i, word in enumerate(word_to_vec.keys()):
        embedding_matrix[i+1] = word_to_vec[word] 

    del word_to_vec
    print("Word embedding loading = done")
    return vocab, word_to_idx, embedding_matrix

def get_embeddings(emb_size:int):
    vocab_path = f'embeddings/vocab_{emb_size}d.txt'
    vectors_path = f'embeddings/embedding_matrix_{emb_size}d.npy'

    with open(vectors_path, 'rb') as f:
        embedding_matrix = np.load(f)

    with open(vocab_path, 'r',  encoding="utf-8") as f:
        vocab = [word.strip() for word in f.readlines()]
    
    return vocab, embedding_matrix