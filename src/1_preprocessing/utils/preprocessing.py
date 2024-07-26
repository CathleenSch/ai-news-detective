import utils.score_calculation as score

import nltk
import numpy as np

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import pad_sequences


training_human_written = []
training_ai_written = []
training_embeddings = []
sentence_burstiness_human = []
sentence_burstiness_ai = []

LENGTH = 300
path = Path(__file__).parent

def read_texts(filepath, paraphrased_filepath):
    print('Reading texts...')
    
    directory = (path / filepath).resolve()
    print(directory)
    files = sorted(directory.glob('*'), key=lambda x: int(x.name))
    
        
    data = []
    for file_path in files:
        with file_path.open('r', encoding='utf-8') as file:
            print(f'reading file: {file_path}')
            data.append(file.read())
    
    if paraphrased_filepath:
        paraphrased_directory = (path / paraphrased_filepath).resolve()
        paraphrased_files = sorted(paraphrased_directory.glob('*'), key=lambda x: int(x.name))
        
        for file_path in paraphrased_files:
            with file_path.open('r', encoding='utf-8') as file:
                print(f'reading file: {file_path}')
                data.append(file.read())
            
    print('Successfully read training data!')
    
    return data
            
def load_word2vec_model():
    print('Loading Word2Vec model...')
    word2vec_path = (path / '../../resources/GoogleNews_Word2Vec/GoogleNews-vectors-negative300.bin').resolve()
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print('Successfully loaded model!')
    return w2v_model

def vectorize_data(data, word2vec_model):
    embeddings = []
    tokenized_data = [word_tokenize(text) for text in data]
    for text in tokenized_data:
        word_embeddings = []
        for token in text:
            if token in word2vec_model:
                word_embeddings.append(word2vec_model[token])
        
        if word_embeddings:
            aggregated_embedding = sum(word_embeddings) / len(word_embeddings)
            embeddings.append(aggregated_embedding)
    
    return embeddings  

def normalize_embeddings(embeddings):
    scaler = MinMaxScaler()
    
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings


def preprocess_data(human_text_filepath, ai_text_filepath, paraphrased_text_filepath, human_output_directory, ai_output_directory):
    nltk.download('punkt')

    ('Reading texts')
    human_data = read_texts(human_text_filepath, None)
    ai_data = read_texts(ai_text_filepath, paraphrased_text_filepath)
    
    model = load_word2vec_model()
    
    print('Vectorizing')
    human_vectorized_data = vectorize_data(human_data, model)
    ai_vectorized_data = vectorize_data(ai_data, model)
    
    human_vectors = score.calculate_and_add_scores(human_data, human_vectorized_data)
    ai_vectors = score.calculate_and_add_scores(ai_data, ai_vectorized_data)

    print('Normalizing embeddings')
    human_normalized_embeddings = normalize_embeddings(human_vectors)
    ai_normalized_embeddings = normalize_embeddings(ai_vectors)
    
    for i, _ in enumerate(ai_normalized_embeddings):
        filename = f'{str(i)}'
        
        if (i < len(human_normalized_embeddings)):
            np.savetxt(human_output_directory + filename, human_normalized_embeddings[i])
            
        np.savetxt(ai_output_directory + filename, ai_normalized_embeddings[i])
    
    return human_normalized_embeddings, ai_normalized_embeddings

