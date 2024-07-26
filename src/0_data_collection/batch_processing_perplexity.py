import json
import numpy as np
import os
import tiktoken as tt

from datetime import datetime
from openai import OpenAI
from pathlib import Path
from utils.openai_utils import read_apikey

MODEL = 'gpt-3.5-turbo'
client = OpenAI(api_key=read_apikey())

def tokenize(text):
    encoding = tt.get_encoding('cl100k_base')
    tokens_numbers = encoding.encode(text)
    byte_tokens = [encoding.decode_single_token_bytes(token) for token in tokens_numbers]
    tokens = [byte_token.decode('utf-8') for byte_token in byte_tokens]
    return tokens

def create_batch_file_perplexity(text_directory, text_type):
    path = Path(__file__).parent
    files = sorted((path / text_directory).resolve().glob('*'), key=lambda x: int(x.name))

    for file in files:
        filepath = Path(file)
        text_no = filepath.stem
        
        with open(file) as f:
            text = f.read()
            
        tokens = tokenize(text)
        batch_file_path = f'./batch_file_{text_type}_{MODEL}_{text_no}'      

        with open(batch_file_path, 'a') as file:
            for i in range(1, len(tokens)):              
                prompt = ' '.join(tokens[:i])
                target_token = tokens[i]
                
                custom_id = f'text_{text_no}-token_{i}_{target_token}'
                if i == 1:
                    print(custom_id)

                task = {
                    'custom_id': custom_id,
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': MODEL,
                        'messages': [
                            {'role': 'user', 'content': prompt}
                        ],
                        'max_tokens': 1,
                        'logprobs': True,
                        'top_logprobs': 20
                    }
                }
                
                file.write(json.dumps(task) + '\n')
        
        with open(batch_file_path, 'rb') as batch_file:
            batch_file_response = client.files.create(
                file = batch_file,
                purpose = 'batch'
            )
        
        batch_job = client.batches.create(
            input_file_id = batch_file_response.id,
            endpoint = '/v1/chat/completions',
            completion_window = '24h'
        )
        
        with open(f'batch_ids_{text_type}_{MODEL}', 'a') as file:
            file.write(f'{datetime.now()} ---> {text_no}\t\t\t{batch_job.id}\n')
            
        os.remove(batch_file_path)
            
def calculate_perplexity(texts, padded_embeddings, directory):
    for text in texts:
        tokens = tokenize(text)
        log_probs = []
        
        for i in range(1, len(tokens)):
            prompt = ' '.join(tokens[:i])
            target_token = tokens[i]
            
            completion = client.chat.completions.create(
                model = MODEL,
                messages = [{'role': 'user', 'content': prompt}],
                max_tokens = 1,
                logprobs = True,
                top_logprobs=20
            )
            
            target_token_found = False
            logprobs_dict = {}
            for choice in completion.choices:
                for top_logprob in choice.logprobs.content[0].top_logprobs:
                    logprobs_dict.update({top_logprob.token: top_logprob.logprob})
                
                if target_token in logprobs_dict:
                    log_prob = logprobs_dict[target_token]
                    log_probs.append(log_prob)
                    target_token_found = True
                    break
            
            default_log_prob = np.log(1e-20)
            if not target_token_found:
                all_log_probs = list(logprobs_dict.values())
                min_log_prob = min(all_log_probs)
                max_log_prob = max(all_log_probs)
                default_log_prob = min(max_log_prob, default_log_prob)
                default_log_prob = max(min_log_prob, default_log_prob)
                log_probs.append(default_log_prob)
        
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)

def add_perplexity_to_embedding(perplexity_values, embeddings):
    return_value = []
    for perplexity, embedding in zip(perplexity_values, embeddings):
        return_value.append(np.append(embedding, perplexity))
    return return_value