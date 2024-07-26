import contractions
import numpy as np
import re
import math

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict

def get_lowercase_tokens(text):
    text = contractions.fix(text)
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens]
    
def calculate_coefficient_of_variation(texts, vectors):
    coefficients = []
    
    for text in texts:
        sentences = re.split(r"(?<!\b\w\.\w)(?<!\b\w\.)(?<=[.!?])\s+", text)
        sentence_lengths = np.array([len(sentence.split()) for sentence in sentences])
        
        sigma = np.std(sentence_lengths)
        mu = np.mean(sentence_lengths)

        cv = sigma / mu if mu != 0 else 0.0
        coefficients.append(cv)
    
    results = add_scores_to_embeddings(coefficients, vectors)
    return results
            
def calculate_burstiness_parameter(texts, vectors):
    burstiness_parameters = []
    
    for text in texts:
        tokens = get_lowercase_tokens(text)
        token_positions = defaultdict(list)
        burstiness_values = []
        
        for i, token in enumerate(tokens):
            token_positions[token].append(i)
            
        
        inter_event_times = []
        for positions in token_positions.values():
            if len(positions) > 1:
                inter_event_times = np.diff(positions)

            mu = np.mean(inter_event_times)
            sigma = np.std(inter_event_times)
            
            burstiness_parameter = (sigma - mu) / (sigma + mu) if (sigma + mu) != 0 else 0  
            burstiness_values.append(burstiness_parameter)

        burstiness_parameters.append(np.mean(burstiness_values))
        
    results = add_scores_to_embeddings(burstiness_parameters, vectors)
    return results

def calculate_herdans_c(texts, vectors):
    herdans_c_scores = []
    for text in texts:
        tokens = get_lowercase_tokens(text)
        
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        
        herdans_c = math.log(unique_tokens) / math.log(total_tokens) if total_tokens > 0 else 0
        herdans_c_scores.append(herdans_c)
    
    results = add_scores_to_embeddings(herdans_c_scores, vectors)
    return results

def calculate_maas(texts, vectors):
    maas_scores = []
    
    for text in texts:
        tokens = get_lowercase_tokens(text)
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        
        if total_tokens > 0:
            maas_score = (math.log(total_tokens) - math.log(unique_tokens)) / math.log(total_tokens)**2
        else:
            maas_score = 0
            
        maas_scores.append(maas_score)
        
    results = add_scores_to_embeddings(maas_scores, vectors)
    return results

def calculate_simpsons_index(texts, vectors):
    simpsons_index_scores = []
    
    for text in texts:
        tokens = get_lowercase_tokens(text)
        total_tokens = len(tokens)
        freq_dist = FreqDist(tokens)
        
        simpsons_index_score = sum(f * (f - 1) for f in freq_dist.values()) / (total_tokens * (total_tokens - 1)) if total_tokens > 1 else 0
        
        simpsons_index_scores.append(simpsons_index_score)

    results = add_scores_to_embeddings(simpsons_index_scores, vectors)
    return results

def add_scores_to_embeddings(scores, vectors):
    vectors = [list(embedding) for embedding in vectors]
    
    for i in range(len(vectors)):
        vectors[i].append(scores[i])
        
    vectors = np.array(vectors)
    
    return vectors

def calculate_and_add_scores(texts, vectors):
    vectors = calculate_coefficient_of_variation(texts, vectors)
    # vectors = calculate_burstiness_parameter(texts, vectors)
    vectors = calculate_herdans_c(texts, vectors)
    vectors = calculate_maas(texts, vectors)
    vectors = calculate_simpsons_index(texts, vectors)
    
    return vectors