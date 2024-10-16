import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Dropout, BatchNormalization
from tensorflow.keras.applications import InceptionResNetV2

def create_base_network(input_shape):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Fine-tune the last few layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    embeddings = Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    normalized_embeddings = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), 
                                   output_shape=(128,))(embeddings)
    
    model = Model(inputs=base_model.input, outputs=normalized_embeddings)
    return model

def create_siamese_network(input_shape):
    base_network = create_base_network(input_shape)
    
    input_anchor = Input(shape=input_shape)
    input_positive = Input(shape=input_shape)
    input_negative = Input(shape=input_shape)
    
    embedding_anchor = base_network(input_anchor)
    embedding_positive = base_network(input_positive)
    embedding_negative = base_network(input_negative)
    
    outputs = tf.keras.layers.Concatenate()([embedding_anchor, embedding_positive, embedding_negative])
    
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=outputs)
    return model