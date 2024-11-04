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
```
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
```
Install dependencies using:
```
pip install tensorflow keras opencv-python mtcnn scikit-learn joblib numpy pandas streamlit pillow requests tempfile fastapi uvicorn
```
## 📂 Data Format & Setup

### Dataset Structure
Before training, ensure your dataset follows this specific structure:
```
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
```
Each class should be in a separate folder containing multiple images of the same person/subject. This structure is crucial for the proper functioning of the training pipeline.

## 🚀 Usage

### Training
To train the model:

```
python train.py
```

Important: Before running the training script, you need to modify the train.py file:
1. Locate the train_model function call
2. Update the data_directory parameter with the path to your dataset:

```
  train_model(data_dir='path/to/your/dataset')
```
### Evaluation
To evaluate model performance:
```
python evaluate.py
```

### Face Recognition
To perform face recognition on images:
```
python recognize.py
```

Note: Before running recognition:
1. Open recognize.py in your preferred editor
2. Locate the image path variable
3. Update it with the path to your test image:
  image_path = 'path/to/your/test/image.jpg'

### Running the Web Interface

1. Start the server:

```
cd Server
python main.py
```

3. In a new terminal, start the client:
```
cd Client
streamlit run client.py
```

The web interface will be accessible through your browser at localhost:8501

## 📁 Project Structure

- data_preprocessing.py: Handles dataset preparation and augmentation
- loss.py: Contains implementation of triplet loss function
- model.py: FaceNet model architecture and Siamese network implementation
- recognize.py: Script for face recognition inference
- train.py: Training script with Siamese network
- evaluate.py: Model evaluation script
- Client/: Contains Streamlit web interface
- Server/: Contains FastAPI server implementation and main.py for server startup

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:
1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request with a comprehensive description of changes

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the original FaceNet paper authors
- MTCNN implementation references
- Siamese Networks research papers
- The open-source community for various dependencies

## 📧 Contact

For any queries or support, please open an issue in the repository.

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Prepare your dataset following the specified structure
4. Run the preprocessing script
5. Train the model with your dataset
6. Start the server and client for the web interface
7. Begin recognizing faces!

For detailed troubleshooting and additional information, please refer to the issues section of the repository.
