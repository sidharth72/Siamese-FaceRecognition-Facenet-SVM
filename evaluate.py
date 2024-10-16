import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import create_siamese_network
from data_preprocessing import create_data_generators

def evaluate_siamese_model(model, val_generator, num_samples = 4):
    base_network = model.layers[3]
    anchors, positives, negatives = [], [], []
    labels = []


    for i in range(min(num_samples, len(val_generator))):
        batch = val_generator[i]
        (a, p, n), _ = batch
        anchors.extend(a)
        positives.extend(p)
        negatives.extend(n)
        labels.extend([i] * len(a))

    anchors = np.array(anchors)
    positives = np.array(positives)
    negatives = np.array(negatives)

    anchor_embeddings = base_network.predict(anchors)
    positive_embeddings = base_network.predict(positives)
    negative_embeddings = base_network.predict(negatives)

    positive_distances = np.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
    negative_distances = np.linalg.norm(anchor_embeddings - negative_embeddings, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(positive_distances, bins=30, alpha=0.5, label='Positive Pairs')
    plt.hist(negative_distances, bins=30, alpha=0.5, label='Negative Pairs')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Positive and Negative Pair Distances')
    plt.legend()
    plt.show()
    
    # Calculate and print average distances
    avg_positive_distance = np.mean(positive_distances)
    avg_negative_distance = np.mean(negative_distances)
    print(f"Average Positive Pair Distance: {avg_positive_distance:.4f}")
    print(f"Average Negative Pair Distance: {avg_negative_distance:.4f}")
    
    # Visualize embeddings using t-SNE if we have enough samples
    all_embeddings = np.vstack([anchor_embeddings, positive_embeddings, negative_embeddings])
    n_samples = all_embeddings.shape[0]
    
    if n_samples >= 50:
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=np.repeat(labels, 3), cmap='viridis')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()
    else:
        print("Not enough samples for t-SNE visualization. Skipping this step.")
    
    # Plot some sample triplets
    num_triplets_to_show = min(15, len(anchors))
    fig, axes = plt.subplots(num_triplets_to_show, 3, figsize=(15, 5 * num_triplets_to_show))
    for i in range(num_triplets_to_show):
        axes[i, 0].imshow(anchors[i])
        axes[i, 0].set_title(f'Anchor')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(positives[i])
        axes[i, 1].set_title(f'Positive\nDist: {positive_distances[i]:.4f}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(negatives[i])
        axes[i, 2].set_title(f'Negative\nDist: {negative_distances[i]:.4f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_dir = 'Data\FaceDatasetNoDuplicates'
    _, val_gen = create_data_generators(data_dir)
    input_shape = (160, 160, 3)
    siamese_model = create_siamese_network(input_shape)
    siamese_model.load_weights('Models/facenet_siamese_finetunedV4.weights.h5')
    evaluate_siamese_model(siamese_model, val_gen, num_samples = 50)