import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.data_utils as du
import yaml 

from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from utils.tqdm_callback import TQDMCallback
from utils.supervised_cl_utils import supervised_contrastive_loss

config = yaml.safe_load(open((Path(__file__).parent / './supervised_config.yml').resolve()))

path = Path(__file__).parent
MODEL_SAVENAME = config['model.type']
FEATURES = config['features']

def build_embedding_model(input_shape):
    input_seq = layers.Input(shape=input_shape)

    reshaped = layers.Reshape((input_shape[0], 1))(input_seq)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(reshaped)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    model = models.Model(inputs=input_seq, outputs=x)
    return model

print('Reading data')
human_train, human_validate, ai_train, ai_validate = du.read_embeddings()

Y_train = []

for _ in human_train:
    Y_train.append(0)
    
for _ in ai_train:
    Y_train.append(1)
    
X_train = np.concatenate([human_train, ai_train], axis=0)
del human_train, ai_train

Y_validate = []

for _ in human_validate:
    Y_validate.append(0)
    
for _ in ai_validate:
    Y_validate.append(1)

X_validate = np.concatenate([human_validate, ai_validate], axis=0)
del human_validate, ai_validate

Y_train = np.array(Y_train)
Y_validate = np.array(Y_validate)

print(f'Training shapes: {X_train.shape} | {Y_train.shape}')
print(f'Validation shapes: {X_validate.shape} | {Y_validate.shape}')

embedding_model = build_embedding_model(input_shape=(FEATURES,)) 

optimizer = optimizers.Adam(learning_rate=0.000001, clipnorm=1.0)
embedding_model.compile(optimizer=optimizer, loss=supervised_contrastive_loss)
embedding_model.summary()

tqdm_callback = TQDMCallback(total_epochs=200)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{path}/logs')

print('Fitting model')
history = embedding_model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_validate, Y_validate), verbose=1, callbacks=[tqdm_callback, early_stopping, tensorboard_callback])
embedding_model.save(f'{path}/model/embedding_model_trained_{MODEL_SAVENAME}.keras')

print('Plotting loss over time')
plt.ioff()

fig = plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model loss over time.png')
plt.close(fig)
