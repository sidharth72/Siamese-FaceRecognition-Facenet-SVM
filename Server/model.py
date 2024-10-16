import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from config import WEIGHTS_PATH, SVM_PATH, CELEBRITY_DICT_PATH, EMBEDDINGS_PATH, INPUT_SHAPE
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

#siamese_model = create_siamese_network(INPUT_SHAPE)

class FaceRecognitionModel:
    def __init__(self):
        self.siamese_network = create_siamese_network(INPUT_SHAPE)
        self.siamese_network.load_weights(WEIGHTS_PATH)
        self.base_network = self.siamese_network.layers[3]
        self.svm = joblib.load(SVM_PATH)
        with open(CELEBRITY_DICT_PATH, 'rb') as f:
            self.celebrity_dict = pickle.load(f)

        embeddings = np.load(EMBEDDINGS_PATH)
        self.scaler = StandardScaler()
        self.scaler.fit(embeddings)

    def recognize_face(self, img):
        img = tf.expand_dims(img, axis = 0)
        embeddings = self.base_network.predict(img)
        normalized_embeddings = self.scaler.transform(embeddings)
        prediction = self.svm.predict(normalized_embeddings)
        return prediction[0]

    def get_celebrity_name(self, prediction):
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()

        
        return self.celebrity_dict.get(prediction, 'Unknown')


