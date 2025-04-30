import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.model_selection import train_test_split
import os
from PIL import Image

def parse_image(filename, label):
    """Parse an image and its label."""
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    """Apply data augmentation to an image."""
    # Random flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Random rotation
    angle = tf.random.uniform((), minval=-AUGMENTATION_PARAMS['rotation_range'], 
                            maxval=AUGMENTATION_PARAMS['rotation_range'])
    image = tf.image.rotate(image, angle * np.pi / 180)
    
    # Random brightness
    image = tf.image.random_brightness(image, AUGMENTATION_PARAMS['brightness_range'])
    
    # Random contrast
    image = tf.image.random_contrast(
        image,
        1 - AUGMENTATION_PARAMS['contrast_range'],
        1 + AUGMENTATION_PARAMS['contrast_range']
    )
    
    # Ensure pixel values are in [0, 1]
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label

def load_data():
    """
    Load and split the data into train, validation, and test sets
    """
    # Load the main train CSV file
    df = pd.read_csv(DATA_DIR / 'aptos2019' / 'train.csv')
    
    # Convert diagnosis to strings
    df['diagnosis'] = df['diagnosis'].astype(str)
    
    # Add .png extension to id_code
    df['id_code'] = df['id_code'] + '.png'
    
    # Split into train, validation, and test sets based on the directory structure
    train_df = df[df['subset'] == 'train']
    val_df = df[df['subset'] == 'validation']
    test_df = df[df['subset'] == 'test']
    
    return train_df, val_df, test_df

def create_data_generators(train_df, val_df):
    """
    Create data generators for training and validation
    """
    print("Creating data generators...")
    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE,
        brightness_range=BRIGHTNESS_RANGE,
        channel_shift_range=30.0
    )

    # Only preprocessing for validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_DIR / 'aptos2019' / 'train_images',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=DATA_DIR / 'aptos2019' / 'train_images',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator

def get_class_weights(train_df):
    """
    Calculate class weights to handle class imbalance
    """
    class_counts = train_df['diagnosis'].value_counts()
    total_samples = len(train_df)
    class_weights = {}
    
    for class_idx in range(NUM_CLASSES):
        class_str = str(class_idx)
        if class_str in class_counts:
            class_weights[class_idx] = total_samples / (NUM_CLASSES * class_counts[class_str])
        else:
            print(f"Warning: No samples found for class {class_idx}")
            class_weights[class_idx] = 1.0
    
    print("Class weights:", class_weights)
    return class_weights

def load_data_from_metadata():
    """
    Load data from metadata.csv, build image paths, and return train, validation, and test dataframes.
    """
    # Load the metadata CSV file
    df = pd.read_csv(DATA_DIR / 'aptos2019' / 'metadata.csv')
    
    # Build the image_path column using the correct columns and convert to string
    df['image_path'] = df.apply(lambda row: str(DATA_DIR / 'aptos2019' / row['folder'] / row['label'] / row['file_name']), axis=1)
    
    # Split the dataframe into train, validation, and test based on the folder
    train_df = df[df['folder'] == 'train']
    val_df = df[df['folder'] == 'validation']
    test_df = df[df['folder'] == 'test']
    
    return train_df, val_df, test_df

def load_data_directly():
    """
    Load data directly from the directory structure without using metadata.csv.
    """
    # Load the train CSV file
    df = pd.read_csv(DATA_DIR / 'aptos2019' / 'train.csv')
    
    # Convert diagnosis to strings
    df['diagnosis'] = df['diagnosis'].astype(str)
    
    # Add .png extension to id_code
    df['id_code'] = df['id_code'] + '.png'
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    # Create data generators for each set with error handling
    def safe_flow_from_dataframe(dataframe, directory, **kwargs):
        try:
            return ImageDataGenerator(rescale=1./255).flow_from_dataframe(
                dataframe=dataframe,
                directory=directory,
                **kwargs
            )
        except Exception as e:
            print(f"Error loading images from {directory}: {e}")
            # Log problematic filenames
            for root, _, files in os.walk(directory):
                for file in files:
                    try:
                        img_path = os.path.join(root, file)
                        with open(img_path, 'rb') as f:
                            img = Image.open(f)
                            img.verify()  # Verify that it is an image
                    except (IOError, SyntaxError) as e:
                        print(f"Problematic file: {img_path} - {e}")
            return None

    train_generator = safe_flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_DIR / 'aptos2019' / 'train_images',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True  # Ensure shuffling for training
    )

    val_generator = safe_flow_from_dataframe(
        dataframe=val_df,
        directory=DATA_DIR / 'aptos2019' / 'train_images',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # No shuffling for validation
    )

    return train_generator, val_generator, None  # Return None for test_generator as it's not used

def load_data_aptos2019():
    """
    Load and split the APTOS 2019 dataset
    """
    # Load the train CSV file
    df = pd.read_csv(DATA_DIR / 'aptos2019' / 'train.csv')
    
    # Convert diagnosis to strings
    df['diagnosis'] = df['diagnosis'].astype(str)
    
    # Add .png extension to id_code
    df['id_code'] = df['id_code'] + '.png'
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=VALIDATION_SPLIT,
        stratify=df['diagnosis'],
        random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    return train_df, val_df 