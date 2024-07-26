import numpy as np
import yaml

from pathlib import Path
from tensorflow.keras.models import load_model
from utils.data_utils import read_embeddings
from utils.supervised_cl_utils import supervised_contrastive_loss

config = yaml.safe_load(open((Path(__file__).parent / './supervised_config.yml').resolve()))

path = Path(__file__).parent
MODEL_TYPE = config['model.type']

print('Loading model')
embedding_model = load_model(f'model/embedding_model_trained_{MODEL_TYPE}.keras', custom_objects={'supervised_contrastive_loss': supervised_contrastive_loss})

human_train, human_validate, ai_train, ai_validate = read_embeddings()

print('Getting human embeddings')
human_train_predicted = embedding_model.predict(human_train)
human_validate_predicted = embedding_model.predict(human_validate)

print('Getting AI embeddings')
ai_train_predicted = embedding_model.predict(ai_train)
ai_validate_predicted = embedding_model.predict(ai_validate)

print('Saving human embeddings')
for i, embedding in enumerate(human_train_predicted):
    filename = f'{path}/../4_classifier/supervised_cl_data/{MODEL_TYPE}/train/human/{i}'
    np.savetxt(filename, embedding)

for i, embedding in enumerate(human_validate_predicted):
    filename = f'{path}/../4_classifier/supervised_cl_data/{MODEL_TYPE}/validate/human/{i}'
    np.savetxt(filename, embedding)

print('Saving ai embeddings')
for i, embedding in enumerate(ai_train_predicted):
    filename = f'{path}/../4_classifier/supervised_cl_data/{MODEL_TYPE}/train/ai/{i}'
    np.savetxt(filename, embedding)

for i, embedding in enumerate(ai_validate_predicted):
    filename = f'{path}/../4_classifier/supervised_cl_data/{MODEL_TYPE}/validate/ai/{i}'
    np.savetxt(filename, embedding)