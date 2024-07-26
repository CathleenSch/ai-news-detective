import joblib
import numpy as np
import yaml

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from tensorflow.keras.models import load_model
from utils.supervised_cl_utils import supervised_contrastive_loss

path = Path(__file__).parent
config = yaml.safe_load(open((Path(__file__).parent / './evaluation_config.yml').resolve()))

CL_MODEL = config['cl.model']
MODEL_TYPE = config['model.type']
DATA_DIRECTORY = config['data.test']

def read_test_embeddings():
    human_embeddings = []
    ai_embeddings = []
    
    print('Reading directories')
    directory_human = (path / f'../../resources/{DATA_DIRECTORY}/test/human').resolve()
    directory_ai = (path / f'../../resources/{DATA_DIRECTORY}/test/ai').resolve()

    files_human = directory_human.glob('*')
    files_ai = directory_ai.glob('*')

    print('Reading human files')
    for file_path in files_human:
        human_embeddings.append(np.loadtxt(file_path))

    print('Reading ai files')
    for file_path in files_ai:
        ai_embeddings.append(np.loadtxt(file_path))
            
    human_embeddings = np.array(human_embeddings)
    ai_embeddings = np.array(ai_embeddings)
            
    return human_embeddings, ai_embeddings

human_embeddings, ai_embeddings = read_test_embeddings()
print(f'Shapes: {human_embeddings.shape} | {ai_embeddings.shape}')

print('Loading siamese model')
embedding_model = load_model(f'./models/{CL_MODEL}/embedding_model_trained_{MODEL_TYPE}.keras', custom_objects={'supervised_contrastive_loss': supervised_contrastive_loss})

generated_human_embeddings = embedding_model.predict(human_embeddings)
del human_embeddings
generated_ai_embeddings = embedding_model.predict(ai_embeddings)
del ai_embeddings

print(f'Shapes: {generated_human_embeddings.shape} | {generated_ai_embeddings.shape}')

X_test = []
Y_test = []

for i in range(generated_human_embeddings.shape[0]):
    X_test.append(generated_human_embeddings[i])
    Y_test.append(0)
    X_test.append(generated_ai_embeddings[i])
    Y_test.append(1)    

del generated_human_embeddings, generated_ai_embeddings

knn_model = joblib.load(f'./models/classifier/{CL_MODEL}_knn_model_{MODEL_TYPE}.pkl')

Y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
mcc = matthews_corrcoef(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print(f'Accuracy score: {accuracy}')
print(f'Matthew\'s correlation coefficient: {mcc}')
print(f'Classification report:')
print(report)