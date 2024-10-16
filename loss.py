import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha=0.2):
    # y_pred is a tensor of shape (batch_size, 3 * embedding_size)
    # We need to reshape it to (batch_size, 3, embedding_size)
    embedding_size = 128  # This should match the size of your embeddings
    y_pred = tf.reshape(y_pred, (-1, 3, embedding_size))
    
    # Now we can split it into anchor, positive, and negative
    anchor = y_pred[:, 0]
    positive = y_pred[:, 1]
    negative = y_pred[:, 2]
    
    # Compute distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # Compute and return triplet loss
    basic_loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(basic_loss)