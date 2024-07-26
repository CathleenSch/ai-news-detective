import keras
import tensorflow.keras.backend as K
import tensorflow as tf

@keras.saving.register_keras_serializable(package="siamese_network_utils", name="contrastive_loss")
def contrastive_loss(y_true, y_pred, margin=1.0):
    print(f'y_true: {y_true.shape} | y_pred: {y_pred.shape}')
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    loss_similar = y_true * tf.square(y_pred)
    loss_dissimilar = (1 - y_true) * tf.square(tf.maximum(0.0, margin - y_pred))
    
    loss = tf.reduce_mean(0.5 * (loss_similar + loss_dissimilar))

    return loss

@keras.saving.register_keras_serializable(package="siamese_network_utils", name="euclidean_distance")
def euclidean_distance(embeddings):
    (embedding_left, embedding_right) = embeddings
    distance = K.sqrt(K.maximum(K.sum(K.square(embedding_left - embedding_right), axis=-1, keepdims=True), K.epsilon()))
    print(f'Distance shape: {distance.shape}')
    return distance

@keras.saving.register_keras_serializable(package="siamese_network_utils", name="euclidean_distance_output_shape")
def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
