import tensorflow as tf
from pathlib import Path
from config import *
from data_loader import load_data_aptos2019, create_data_generators, get_class_weights
from model import create_model, unfreeze_model
from train import train_model, evaluate_model, plot_training_curves
from visualization import visualize_gradcam, get_last_conv_layer_name
import os

def main():
    try:
        # Set random seeds for reproducibility
        tf.random.set_seed(RANDOM_SEED)
        
        # Create necessary directories
        for dir_path in [MODEL_SAVE_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
            dir_path.mkdir(exist_ok=True)
        
        # Load and prepare data using the APTOS 2019 dataset
        print("Loading data...")
        train_df, val_df = load_data_aptos2019()
        
        print("Creating data generators...")
        train_generator, val_generator = create_data_generators(train_df, val_df)
        
        # Calculate class weights
        print("Calculating class weights...")
        class_weights = get_class_weights(train_df)
        
        # Train the model
        print("\nTraining model...")
        model, history = train_model(train_generator, val_generator, class_weights)
        
        # Evaluate the model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, val_generator, history)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
