import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = 'Models/facenet_siamese_finetunedV4.weights.h5'
SVM_PATH = 'Models/svm_classifierV2.joblib'
CELEBRITY_DICT_PATH = 'Data/celebrity_dict2.pkl'
EMBEDDINGS_PATH = 'Data/embeddings.npy'

# Model Configuration
INPUT_SHAPE = (160, 160, 3)
