import joblib
import numpy as np
import utils.calculate_scores as cs
import yaml
 
from gensim.models import KeyedVectors
from pathlib import Path
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time
from utils.supervised_cl_utils import supervised_contrastive_loss

path = Path(__file__).parent
config = yaml.safe_load(open((Path(__file__).parent / './evaluation_config.yml').resolve()))

CL_MODEL = config['cl.model']
MODEL_TYPE = config['model.type']
DATA_DIRECTORY = config['data.test']

def evaluate(texts):
    time_start_pp = time()
    texts_preprocessed = preprocess(texts)
    time_end_pp = time()
    time_pp = time_end_pp - time_start_pp
    
    embedding_model = load_model(f'./models/{CL_MODEL}/embedding_model_trained_{MODEL_TYPE}.keras', custom_objects={'supervised_contrastive_loss': supervised_contrastive_loss})
    time_start_get_embedding = time()
    print('Getting embeddings from model')
    embeddings = embedding_model.predict(texts_preprocessed)
    time_end_get_embedding = time()
    time_embedding = time_end_get_embedding - time_start_get_embedding
    labels = np.ones(embeddings.shape[0])

    knn_model = joblib.load(f'./models/classifier/{CL_MODEL}_knn_model_{MODEL_TYPE}.pkl')
    
    time_start_predict = time()
    print('Predicting...')
    Y_pred = knn_model.predict(embeddings)
    time_end_predict = time()
    time_predict = time_end_predict - time_start_predict
    
    time_total = time_pp + time_embedding + time_predict
    
    accuracy = accuracy_score(labels, Y_pred)
    mcc = matthews_corrcoef(labels, Y_pred)
    report = classification_report(labels, Y_pred)

    print(f'Accuracy score: {accuracy}')
    print(f'Matthew\'s correlation coefficient: {mcc}')
    print(f'Classification report:')
    print(report)
    
    return Y_pred, time_total

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
    
    result = pad_sequences(vectorized, maxlen=300, padding='post', truncating='post', value=0, dtype='float64')
    return result

texts = []

directory = (path / f'../../resources/paraphrased/').resolve()
files = directory.glob('*')

for file_path in files:
    file = file_path.open('r', encoding='utf-8')
    texts.append(file.read())

evaluate(texts)
