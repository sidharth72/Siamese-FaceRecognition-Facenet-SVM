import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model import create_siamese_network
import joblib
import pickle

def preprocess_image(image_path, target_size=(160, 160)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img

def recognize_face(image_path, base_network, classifier, scaler):
    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis = 0)
    embeddings = base_network.predict(img)
    normalized_embeddings = scaler.transform(embeddings)
    prediction = classifier.predict(normalized_embeddings)
    return prediction[0]

if __name__ == '__main__':
    embedding_path = 'Data/embeddings.npy'
    labels_path = 'Data/labels.npy'
    weights_path = 'Models/facenet_siamese_finetunedV4.weights.h5'
    svm_path = "Data/svm_classifierV2.joblib"
    celebrity_dict_path = "Data\celebrity_dict2.pkl"

    # Load embeddings and labels
    embeddings = np.load(embedding_path)
    labels = np.load(labels_path)

    # Create and load the base network
    input_shape = (160, 160, 3)
    siamese_network  = create_siamese_network(input_shape)
    siamese_network.load_weights(weights_path)
    base_network = siamese_network.layers[3]

    # Load the SVM classifier
    svm = joblib.load(svm_path)

    
    with open(celebrity_dict_path, 'rb') as f:
        celebrity_dict = pickle.load(f)

    scaler = StandardScaler()
    scaler.fit(embeddings)

    image_path = "Data/FaceDatasetNoDuplicates/Natalie Portman/003_13b7bb9d_aug_0.jpg"

    # Perform face recognition
    prediction = recognize_face(image_path, base_network, svm, scaler)

    if isinstance(prediction, np.ndarray):
        prediction = prediction.item()
    if prediction in celebrity_dict:
        print("Recognized face:", celebrity_dict[prediction])
    else:
        print("Unrecognized face. Prediction Value:", prediction)