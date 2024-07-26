from openai import OpenAI
from utils.openai_utils import read_apikey

BATCH_JOB_ID = '' # insert batch id here
client = OpenAI(api_key=read_apikey())

batch_job = client.batches.retrieve(BATCH_JOB_ID)
print(batch_job)