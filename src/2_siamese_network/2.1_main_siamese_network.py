import numpy as np
import tensorflow as tf
import utils.data_utils as du
import yaml

from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from utils.siamese_network_utils import contrastive_loss, euclidean_distance, euclidean_distance_output_shape
from utils.tqdm_callback import TQDMCallback

config = yaml.safe_load(open((Path(__file__).parent / './siamese_network_config.yml').resolve()))

path = Path(__file__).parent
MODEL_TYPE = config['model.type']
FEATURES = config['features']
DATA_DIRECTORY = config['data.directory']

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

def build_siamese_model(input_shape):
    input_left = layers.Input(shape=input_shape)
    input_right = layers.Input(shape=input_shape)
    
    embedding_model = build_embedding_model(input_shape)
    
    output_left = embedding_model(input_left)
    output_right = embedding_model(input_right)
    
    distance = layers.Lambda(euclidean_distance, output_shape=euclidean_distance_output_shape)([output_left, output_right])
    
    siamese_model = models.Model(inputs=[input_left, input_right], outputs=distance)
    
    return siamese_model, embedding_model

print('Reading data')
human_train, human_validate, ai_train, ai_validate = du.read_embeddings()

print('Creating training pairs')
training_pairs, Y_train = du.create_training_pairs(human_train, ai_train, 22400)
validation_pairs, Y_validate = du.create_training_pairs(human_validate, ai_validate, 4800)

X_train_1 = np.array([pair[0] for pair in training_pairs])
X_train_2 = np.array([pair[1] for pair in training_pairs])
del training_pairs

print(f'Shape of training pairs: {X_train_1.shape} | {X_train_2.shape}')

X_validate_1 = np.array([pair[0] for pair in validation_pairs])
X_validate_2 = np.array([pair[1] for pair in validation_pairs])
del validation_pairs

print(f'Shape of validation pairs: {X_validate_1.shape} | {X_validate_2.shape}')

siamese_model, embedding_model = build_siamese_model(input_shape=(FEATURES,)) 

optimizer = optimizers.Adam(learning_rate=0.000001, clipnorm=1.0)
siamese_model.compile(optimizer=optimizer, loss=contrastive_loss)
siamese_model.summary()

tqdm_callback = TQDMCallback(total_epochs=200)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{path}/logs')

print('Fitting siamese model')
history = siamese_model.fit([X_train_1, X_train_2], Y_train, epochs=200, batch_size=32, validation_data=([X_validate_1, X_validate_2], Y_validate), verbose=1, callbacks=[tqdm_callback, early_stopping, tensorboard_callback])
siamese_model.save(f'{path}/model/siamese_model_{MODEL_TYPE}.keras')
embedding_model.save(f'{path}/model/embedding_model_{MODEL_TYPE}.keras')
