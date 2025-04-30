import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_SAVE_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
VISUALIZATION_DIR = RESULTS_DIR / 'visualizations'

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data parameters
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced from 32 for better gradient updates
NUM_CLASSES = 5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15
RANDOM_SEED = 42
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Training parameters
INITIAL_LEARNING_RATE = 1e-4  # Initial learning rate for top layers
PHASE2_LEARNING_RATE = 1e-6  # Very low learning rate for fine-tuning as requested
PHASE1_EPOCHS = 15  # Training with frozen layers
PHASE2_EPOCHS = 15  # Fine-tuning phase
EARLY_STOPPING_PATIENCE = 3  # Quick exit if no improvement
LR_PATIENCE = 2  # Adjust learning rate quickly if needed
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model parameters
MODEL_NAME = 'densenet121'
TRAINABLE_LAYERS = 40  # Number of trainable layers from the end
DENSE_UNITS = 2048
DROPOUT_RATE = 0.4
L2_LAMBDA = 0.0005

# Data augmentation parameters - increased for stronger augmentation
ROTATION_RANGE = 45  # Increased rotation range
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.3  # Increased zoom range
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'
BRIGHTNESS_RANGE = [0.6, 1.4]  # Increased brightness range
CONTRAST_RANGE = [0.6, 1.4]  # Increased contrast range

# Data augmentation dictionary
AUGMENTATION_PARAMS = {
    'rotation_range': ROTATION_RANGE,
    'width_shift_range': WIDTH_SHIFT_RANGE,
    'height_shift_range': HEIGHT_SHIFT_RANGE,
    'shear_range': SHEAR_RANGE,
    'zoom_range': ZOOM_RANGE,
    'horizontal_flip': HORIZONTAL_FLIP,
    'fill_mode': FILL_MODE,
    'brightness_range': BRIGHTNESS_RANGE,
    'contrast_range': CONTRAST_RANGE
}

# Dataset paths
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

# Model checkpoint parameters
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_ONLY = True
MONITOR_METRIC = 'val_accuracy'

# Tensorboard parameters
TENSORBOARD_UPDATE_FREQ = 'epoch'

# Image preprocessing
NUM_WORKERS = 4 