import tensorflow as tf
from data_preprocessing import create_data_generators
from models import create_siamese_network
from loss import triplet_loss
import os

def train_model(
    data_dir,
    input_shape=(160, 160, 3),
    batch_size=32,
    epochs=20,
    version="v1"):
    try:
        # Create data generators
        train_gen, val_gen = create_data_generators(data_dir, batch_size)
        
        # Create and compile the model
        siamese_model = create_siamese_network(input_shape)
        siamese_model.compile(optimizer='adam', loss=triplet_loss)
        
        # Set up checkpointing
        checkpoint_path = f'checkpoints/facenet_siamese_finetuned{version}' + '/cp-{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        
        # Train the model
        history = siamese_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[checkpoint_callback]
        )
        
        # Save the final weights
        final_weights_path = f'Models/facenet_siamese_finetuned{version}.weights.h5'
        siamese_model.save_weights(final_weights_path)
        print(f"Final weights saved to {final_weights_path}")
        
        return siamese_model, history
    
    except FileNotFoundError as e:
        print(f"Error: Data directory not found. {e}")
        return None, None
    except tf.errors.OpError as e:
        print(f"TensorFlow error occurred: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    
