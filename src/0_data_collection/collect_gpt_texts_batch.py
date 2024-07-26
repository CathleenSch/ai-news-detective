import json

from datetime import datetime
from openai import OpenAI
from pathlib import Path
from utils.openai_utils import read_apikey

model = 'gpt-3.5-turbo'
client = OpenAI(api_key=read_apikey())

def get_articles_batch():
    batch_file_patch = str(Path(__file__).parent / ('../../resources/batch_file_' + model))
    
    batch_file = client.files.create(
        file = open(batch_file_patch, 'rb'),
        purpose = 'batch'
    )
    
    batch_job = client.batches.create(
        input_file_id = batch_file.id,
        endpoint = '/v1/chat/completions',
        completion_window = '24h'
    )
    
    with open('batch_id', 'a') as file:
        file.write(f'{datetime.now()} ---> {batch_job.id}\n')
            
def create_batch_file():
    tasks = []
    
    for i in range(32000):
        task = {
            'custom_id': f'{i}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': model,
                'messages': [
                        {'role': 'user', 
                         'content': f'Write a short newspaper article (no more than 500 words) without title, headline and metadata and without placeholders. Make it as specific and realistic as possible.'
                        }
                    ],
                'temperature': 1.3
            }
        }
        
        tasks.append(task)
    
    directory = str(Path(__file__).parent / '../../resources/')
    with open(f'{directory}/batch_file_' + model, 'w') as file:
        for task in tasks:
            file.write(json.dumps(task) + '\n')    
        
create_batch_file()
get_articles_batch()