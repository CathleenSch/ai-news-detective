import numpy as np
import random
import yaml

from pathlib import Path

config = yaml.safe_load(open((Path(__file__).parent / './siamese_network_config.yml').resolve()))

path = Path(__file__).parent
DATA_DIRECTORY = config['data.directory']

def read_embeddings():
    human_train = []
    human_validate = []
    ai_train = []
    ai_validate = []
    
    directory_human_train = (path / f'../../resources/{DATA_DIRECTORY}/train/human/').resolve()
    directory_human_validate = (path / f'../../resources/{DATA_DIRECTORY}/validate/human/').resolve()
    directory_ai_train = (path / f'../../resources/{DATA_DIRECTORY}/train/ai/').resolve()
    directory_ai_validate = (path / f'../../resources/{DATA_DIRECTORY}/validate/ai/').resolve()


    files_human_train = directory_human_train.glob('*')
    files_human_validate = directory_human_validate.glob('*')
    files_ai_train = directory_ai_train.glob('*')
    files_ai_validate = directory_ai_validate.glob('*')

    print('Reading human training files')
    for file_path in files_human_train:
        human_train.append(np.loadtxt(file_path))
    
    print('Reading human validation files')
    for file_path in files_human_validate:
        human_validate.append(np.loadtxt(file_path))
    
    print('Reading ai training files')
    for file_path in files_ai_train:
        ai_train.append(np.loadtxt(file_path))
    
    print('Reading ai validation files')
    for file_path in files_ai_validate:
        ai_validate.append(np.loadtxt(file_path))

    human_train = np.array(human_train)
    human_validate = np.array(human_validate)
    ai_train = np.array(ai_train)
    ai_validate = np.array(ai_validate)
    
    print(f'Shapes: {human_train.shape} | {human_validate.shape} || {ai_train.shape} | {ai_validate.shape}')
    
    return human_train, human_validate, ai_train, ai_validate
    
def create_training_pairs(human_embeddings, ai_embeddings, iterations):
    training_pairs = []
    labels = []

    for x in range(0, iterations): 
        if x % 1000 == 0:
            print(f'Round {x} of creating training pairs')
        training_pairs.append([random.choice(human_embeddings), random.choice(human_embeddings)])
        labels.append(1)
        
        training_pairs.append([random.choice(human_embeddings), random.choice(ai_embeddings)])
        labels.append(0)
        
        training_pairs.append([random.choice(ai_embeddings), random.choice(ai_embeddings)])
        labels.append(1)
        
        training_pairs.append([random.choice(ai_embeddings), random.choice(human_embeddings)])
        labels.append(0)

    return np.array(training_pairs), np.array(labels)