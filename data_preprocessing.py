import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import defaultdict

class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, img_size=(160, 160), is_training=True, val_split=0.2, triplets_per_anchor=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.triplets_per_anchor = triplets_per_anchor
        
        self.image_paths, self.labels = self.get_image_paths_and_labels()
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        
        self.train_indices, self.val_indices = train_test_split(
            range(len(self.image_paths)), test_size=val_split, stratify=self.labels, random_state=42
        )
        
        self.current_indices = self.train_indices if is_training else self.val_indices
        np.random.shuffle(self.current_indices)
        
        self.image_cache = {}

    def get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for label, celebrity in enumerate(os.listdir(self.data_dir)):
            celebrity_dir = os.path.join(self.data_dir, celebrity)
            if os.path.isdir(celebrity_dir):
                for img in os.listdir(celebrity_dir):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(celebrity_dir, img))
                        labels.append(label)
        return np.array(image_paths), np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.current_indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.current_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        anchors, positives, negatives = [], [], []
        
        for index in batch_indices:
            anchor_img = self.image_paths[index]
            anchor_label = self.labels[index]
            
            for _ in range(self.triplets_per_anchor):
                positive_index = np.random.choice([i for i in self.label_to_indices[anchor_label] if i != index])
                positive_img = self.image_paths[positive_index]
                
                negative_label = np.random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
                negative_index = np.random.choice(self.label_to_indices[negative_label])
                negative_img = self.image_paths[negative_index]
                
                anchors.append(self.preprocess_image(anchor_img))
                positives.append(self.preprocess_image(positive_img))
                negatives.append(self.preprocess_image(negative_img))
        
        return (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros((len(anchors),))

    def preprocess_image(self, img_path):
        if img_path in self.image_cache:
            return self.image_cache[img_path]
        
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
        
        self.image_cache[img_path] = img
        return img

    def on_epoch_end(self):
        np.random.shuffle(self.current_indices)
        self.image_cache.clear()  # Clear cache at the end of each epoch

def create_data_generators(data_dir, batch_size=32):
    train_gen = TripletDataGenerator(data_dir, batch_size=batch_size, is_training=True)
    val_gen = TripletDataGenerator(data_dir, batch_size=batch_size, is_training=False)
    return train_gen, val_gen