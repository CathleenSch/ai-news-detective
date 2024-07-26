import json

from datetime import datetime
from openai import OpenAI
from pathlib import Path
from utils.openai_utils import read_apikey

model = 'gpt-3.5-turbo'
client = OpenAI(api_key=read_apikey())
path = Path(__file__).parent

def create_batch_file(texts):
    tasks = []
    
    for i, text in enumerate(texts):
        task = {
            'custom_id': f'{i}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': model,
                'messages': [
                        {'role': 'user', 
                         'content': f'Please paraphrase the following text: {text}'
                        }
                    ],
                'temperature': 1.3
            }
        }
        
        tasks.append(task)
    
    directory = str(Path(__file__).parent / '../../resources/')
    with open(f'{directory}/batch_file_paraphrase' + model, 'w') as file:
        for task in tasks:
            file.write(json.dumps(task) + '\n')    


def get_articles_batch():
    batch_file_patch = str(Path(__file__).parent / ('../../resources/batch_file_paraphrase_' + model))
    
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

print('Reading texts...')
    
directory = (path / '../../resources/gpt-3.5/').resolve()
print(directory)
files = sorted(directory.glob('*'), key=lambda x: int(x.name))
data = []
for file_path in files:
    with file_path.open('r', encoding='utf-8') as file:
        print(f'reading file: {file_path}')
        data.append(file.read())

print('Successfully read data!')

create_batch_file(data)
get_articles_batch()