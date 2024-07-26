import utils.calculate_scores as cs
import joblib
 
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from utils.supervised_cl_utils import supervised_contrastive_loss
from tensorflow.keras.models import load_model

path = Path(__file__).parent

def detect(texts):
    texts_preprocessed = preprocess(texts)

    embedding_model = load_model(f'./models/supervised_cl/embedding_model_trained_304_complex.keras', custom_objects={'supervised_contrastive_loss': supervised_contrastive_loss})

    print('Getting embeddings from model')
    embeddings = embedding_model.predict(texts_preprocessed)

    knn_model = joblib.load(f'./models/classifier/supervised_cl_knn_model_304_complex.pkl')
    
    Y_pred = knn_model.predict(embeddings)
    
    return Y_pred

def preprocess(texts):
    vectors = vectorize_texts(texts)
    print('Calculating Scores')
    vectors = cs.calculate_and_add_scores(texts, vectors)

    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(vectors)
    
    return normalized_vectors

def vectorize_texts(texts):
    print('Loading Word2Vec model...')
    w2v_model = KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
    print('Successfully loaded model!')
    
    vectorized = []
    tokenized_data = [word_tokenize(text) for text in texts]
    for text in tokenized_data:
        word_embeddings = []
        for token in text:
            if token in w2v_model:
                word_embeddings.append(w2v_model[token])
        
        if word_embeddings:
            aggregated_embedding = sum(word_embeddings) / len(word_embeddings)
            vectorized.append(aggregated_embedding)
    
    return vectorized

print('bla')