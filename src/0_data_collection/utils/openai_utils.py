import openai
import yaml

from pathlib import Path

def read_apikey():
    config = yaml.safe_load(open((Path(__file__).parent / '../../config_local.yml').resolve()))
    return config['apikey']

def set_openai_apikey():
    openai.api_key = read_apikey()