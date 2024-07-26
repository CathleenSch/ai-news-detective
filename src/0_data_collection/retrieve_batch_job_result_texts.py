import json

from openai import OpenAI
from pathlib import Path
from utils.openai_utils import read_apikey

BATCH_JOB_ID = 'batch_XbBHUIHxRncMvtfDxOQiqKbp' # change this id corresponding to the batch that was created
client = OpenAI(api_key=read_apikey())

batch_job = client.batches.retrieve(BATCH_JOB_ID)
result_file_id = batch_job.output_file_id
batch_job_result = client.files.content(result_file_id).content

directory = str(Path(__file__).parent / '../../resources/paraphrased/')
with open(f'{directory}/batch_job_result', 'wb') as file:
    file.write(batch_job_result)

results = []
with open(f'{directory}/batch_job_result', 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        results.append(json_object)

for result in results:
    index = int(result['custom_id'].split('-')[-1])
    content = result['response']['body']['choices'][0]['message']['content']
    id = result['response']['body']['id']
    filename = f'{index}'

    with open(f'{directory}/{filename}', 'w', encoding='utf-8') as file:
        file.write(content)