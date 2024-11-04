# FaceNet Siamese Face Recognition

A robust face recognition system built using FaceNet architecture trained on Siamese Framework. The system can identify and classify faces across 16 different celebrity classes with high accuracy.

## 🎯 Features

- Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Face recognition using FaceNet architecture
- Siamese Network training framework with triplet loss
- Real-time face recognition capabilities
- Web interface using Streamlit
- REST API server using FastAPI
- Support for custom dataset training

## 🏗️ Architecture

The system uses a combination of powerful deep learning models:

1. **MTCNN**: For accurate face detection and alignment
2. **FaceNet**: Pre-trained model modified and fine-tuned for feature extraction
3. **Siamese Network**: Training framework using triplet loss (anchor, positive, negative samples)

## 🛠️ Requirements

### Required Modules
tensorflow
keras
opencv-python
mtcnn
scikit-learn
joblib
numpy
pandas
streamlit
pillow
requests
tempfile
fastapi
uvicorn

Install dependencies using:
pip install -r requirements.txt

## 📂 Data Format

For training with custom dataset, organize your data in the following structure:
dataset/
    ├── Class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── Class2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...

## 🚀 Usage

### Data Preprocessing
To prepare your dataset:
python data_preprocessing.py

### Training
To train the model:
python train.py

### Evaluation
To evaluate model performance:
python evaluate.py

### Face Recognition
To perform face recognition on images:
python recognize.py

### Running the Web Interface

1. Start the server:
cd Server
python server.py

2. In a new terminal, start the client:
cd Client
streamlit run client.py

## 📁 Project Structure

- data_preprocessing.py: Handles dataset preparation and augmentation
- loss.py: Contains implementation of triplet loss function
- model.py: FaceNet model architecture and Siamese network implementation
- recognize.py: Script for face recognition inference
- train.py: Training script with Siamese network
- evaluate.py: Model evaluation script
- Client/: Contains Streamlit web interface
- Server/: Contains FastAPI server implementation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the original FaceNet paper authors
- MTCNN implementation references
- Siamese Networks research papers

## 📧 Contact

For any queries or support, please open an issue in the repository.
