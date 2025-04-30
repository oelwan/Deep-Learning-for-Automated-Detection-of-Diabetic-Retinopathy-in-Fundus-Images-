import os
import tensorflow as tf
from model import create_model
from data_loader import load_data_aptos2019, create_data_generators

print("Testing DenseNet121 model setup...")

# Test model creation
model = create_model()
print(f"Model created successfully with {model.count_params()} parameters")
print(f"Base model: {model.layers[1].__class__.__name__}")

# Test data loading
train_df, val_df = load_data_aptos2019()
print(f"Data loaded successfully: {len(train_df)} training samples, {len(val_df)} validation samples")

# Test data generator creation
print("Creating data generators...")
train_generator, val_generator = create_data_generators(train_df, val_df)
print("Data generators created successfully")

print("\nRESULT: All setup tests passed! The code is ready to train the DenseNet121 model.") 