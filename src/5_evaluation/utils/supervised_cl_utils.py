import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(package="supervised_cl_utils", name="supervised_contrastive_loss")
def supervised_contrastive_loss(y_true, y_pred, temperature=0.1):
    y_true = tf.cast(y_true, tf.float32)

    # normalize vectors for cosine similarity
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    
    # calculate dot product for each combination (result is cosine similarity due to previous normalization)
    similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)
    
    # create filter for positive samples
    labels_expanded = tf.expand_dims(y_true, 0) == tf.expand_dims(y_true, 1)
    labels_expanded = tf.cast(labels_expanded, tf.float32) - tf.eye(tf.shape(y_true)[0])
    
    # calculate exponential similarities for all pairs
    exp_sim_matrix = tf.exp(similarity_matrix / temperature)
    
    # calculate sum of exponential similarities for all pairs
    total_exp = tf.reduce_sum(exp_sim_matrix, axis=1, keepdims=True)
    
    # calculate exponential similarities for positive pairs by applying the filter
    positive_exp = exp_sim_matrix * labels_expanded + 1e-12
    
    # make sure that division by zero can't happen
    total_exp = tf.maximum(total_exp, 1e-12)
    
    # calculate individual log losses for positive pairs
    log_prob = tf.math.log(positive_exp / total_exp)
    
    # sum the log probabilities for positive pairs and divide by number of positive samples
    loss = -tf.reduce_sum(labels_expanded * log_prob) / tf.reduce_sum(labels_expanded)
    
    return loss
