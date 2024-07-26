import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

DIRECTORY = 'modeldata_no_burstiness'
path = Path(__file__).parent

human_embeddings = []
ai_embeddings = []

print('Reading directories')
directory_human_embeddings = (path / './human_preprocessed/').resolve()
directory_human_train = f'../../resources/{DIRECTORY}/train/human/'
directory_human_validate = f'../../resources/{DIRECTORY}/validate/human/'
directory_human_test = f'../../resources/{DIRECTORY}/test/human/'

directory_ai_embeddings = (path / './ai_preprocessed/').resolve()
directory_ai_train = f'../../resources/{DIRECTORY}/train/ai/'
directory_ai_validate = f'../../resources/{DIRECTORY}/validate/ai/'
directory_ai_test = f'../../resources/{DIRECTORY}/test/ai/'

files_human = sorted(directory_human_embeddings.glob('*'), key = lambda x: x.name)
files_ai = sorted(directory_ai_embeddings.glob('*'), key = lambda x: x.name)

print('Reading human files')
for file_path in files_human:
    human_embeddings.append(np.loadtxt(file_path))

human_embeddings = np.array(human_embeddings)

print('Reading ai files')
for file_path in files_ai:
    ai_embeddings.append(np.loadtxt(file_path))

ai_embeddings = np.array(ai_embeddings)

print(f'Shapes: {human_embeddings.shape} | {ai_embeddings.shape}')

print('Splitting human data')    
human_train, human_temp = train_test_split(human_embeddings, test_size=0.3)
del human_embeddings
human_validate, human_test = train_test_split(human_temp, test_size=0.5)
del human_temp

print('Splitting ai data')
ai_train, ai_temp = train_test_split(ai_embeddings, test_size=0.3)
del ai_embeddings
ai_validate, ai_test = train_test_split(ai_temp, test_size=0.5)
del ai_temp

print('Writing training data')
for i, _ in enumerate(human_train):
    np.savetxt(directory_human_train + str(i), human_train[i])
    np.savetxt(directory_ai_train + str(i), ai_train[i])
    
print('Writing validation data')
for i, _ in enumerate(human_validate):
    np.savetxt(directory_human_validate + str(i), human_validate[i])
    np.savetxt(directory_ai_validate + str(i), ai_validate[i])

print('Writing test data')
for i, _ in enumerate(human_test):
    np.savetxt(directory_human_test + str(i), human_test[i])
    np.savetxt(directory_ai_test + str(i), ai_test[i])