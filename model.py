import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from config import *

def create_model():
    # Load the base model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    
    # Add custom layers with improved architecture
    x = GlobalAveragePooling2D()(x)
    
    # First dense block
    x = Dense(DENSE_UNITS, activation='relu', 
              kernel_regularizer=l2(L2_LAMBDA))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Second dense block
    x = Dense(DENSE_UNITS // 2, activation='relu',
              kernel_regularizer=l2(L2_LAMBDA))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Final dense block
    x = Dense(DENSE_UNITS // 4, activation='relu',
              kernel_regularizer=l2(L2_LAMBDA))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model with improved optimizer settings
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE, 
                      beta_1=0.9, 
                      beta_2=0.999, 
                      epsilon=1e-07),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model(model):
    """Unfreeze the last TRAINABLE_LAYERS layers of the model for fine-tuning."""
    # Unfreeze the base model
    model.trainable = True
    
    # Get total number of layers in DenseNet121 base
    total_layers = len(model.layers[1].layers)
    
    # Calculate which layer to start unfreezing from
    fine_tune_at = total_layers - TRAINABLE_LAYERS
    
    # Make sure we don't go below 0
    fine_tune_at = max(0, fine_tune_at)
    
    # Freeze all layers except the last TRAINABLE_LAYERS
    for layer in model.layers[1].layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile the model with new learning rate
    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LEARNING_RATE,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-07),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 