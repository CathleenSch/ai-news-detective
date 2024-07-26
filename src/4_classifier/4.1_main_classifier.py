import numpy as np
import tqdm as tqdm

from pathlib import Path
from utils.classifiers import train_model
from utils.data_utils import read_embeddings

path = Path(__file__).parent

human_train, human_validate, ai_train, ai_validate = read_embeddings()
human_train_labels = np.zeros(human_train.shape[0])
human_validate_labels = np.zeros(human_validate.shape[0])
ai_train_labels = np.ones(ai_train.shape[0])
ai_validate_labels = np.ones(ai_validate.shape[0])

X_train = np.concatenate((human_train, ai_train), axis=0)
del human_train, ai_train
Y_train = np.concatenate((human_train_labels, ai_train_labels), axis=0)
del human_train_labels, ai_train_labels

X_validate = np.concatenate((human_validate, ai_validate), axis=0)
del human_validate, ai_validate
Y_validate = np.concatenate((human_validate_labels, ai_validate_labels), axis=0)
del human_validate_labels, ai_validate_labels

print(f'Shapes: {X_train.shape} | {Y_train.shape} || {X_validate.shape} | {Y_validate.shape}')

train_model('knn', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
# train_model('mlp', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
# train_model('svm', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
# train_model('rf', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
# train_model('lr', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
# train_model('gb', X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)

