import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import *
from model import create_model, unfreeze_model
from data_loader import create_data_generators, load_data, load_data_directly, load_data_aptos2019
import tensorflow as tf

class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(HistoryCallback, self).__init__()
        self.logs = []
        self.phase = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Ensure all log values are numeric
            numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            numeric_logs['phase'] = self.phase
            self.logs.append(numeric_logs)

def create_callbacks(phase, model_name):
    """
    Create callbacks for training
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = MODEL_SAVE_DIR / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    log_dir = RESULTS_DIR / 'logs' / model_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / f'model_{phase}_epoch{{epoch:02d}}_val_acc{{val_accuracy:.4f}}.h5'),
            monitor=MONITOR_METRIC,
            mode='max',
            save_best_only=SAVE_BEST_ONLY,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        HistoryCallback()
    ]
    
    return callbacks

def get_class_weights(dataframe):
    """
    Calculate class weights to handle class imbalance
    """
    # Get the class counts from the dataframe
    class_counts = dataframe['diagnosis'].value_counts()
    total_samples = len(dataframe)
    num_classes = len(class_counts)
    class_weights = {}
    
    for class_idx in range(num_classes):
        class_str = str(class_idx)
        if class_str in class_counts:
            class_weights[class_idx] = total_samples / (num_classes * class_counts[class_str])
        else:
            print(f"Warning: No samples found for class {class_idx}")
            class_weights[class_idx] = 1.0
    
    print("Class weights:", class_weights)
    return class_weights

def train_model(train_generator, val_generator, class_weights):
    """
    Train the model using a two-phase strategy
    """
    # Create the model
    model = create_model()
    
    # Phase 1: Train only the top layers
    print("\nPhase 1: Training top layers...")
    phase1_callbacks = create_callbacks('phase1', MODEL_NAME)
    phase1_callbacks[-1].phase = 'phase1'
    
    history1 = model.fit(
        train_generator,
        epochs=PHASE1_EPOCHS,
        validation_data=val_generator,
        callbacks=phase1_callbacks,
        class_weight=class_weights,
        batch_size=BATCH_SIZE
    )
    
    # Phase 2: Fine-tune the entire model
    print("\nPhase 2: Fine-tuning entire model...")
    model = unfreeze_model(model)
    phase2_callbacks = create_callbacks('phase2', MODEL_NAME)
    phase2_callbacks[-1].phase = 'phase2'
    
    history2 = model.fit(
        train_generator,
        epochs=PHASE2_EPOCHS,
        validation_data=val_generator,
        callbacks=phase2_callbacks,
        class_weight=class_weights,
        batch_size=BATCH_SIZE
    )
    
    # Combine histories
    combined_history = {
        'phase1': history1.history,
        'phase2': history2.history
    }
    
    return model, combined_history

def evaluate_model(model, test_generator, history=None):
    """
    Evaluate the model and save results
    """
    # Get predictions
    y_pred_proba = model.predict(test_generator)
    y_true = test_generator.classes

    # Convert y_true to numpy array if it's a list
    y_true = np.array(y_true)
    
    # Get class predictions for other metrics
    y_pred = y_pred_proba.argmax(axis=1)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    }
    
    # Save metrics
    metrics_path = RESULTS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(VISUALIZATION_DIR / 'confusion_matrix.png')
    plt.close()
    
    # If history is provided, save it
    if history is not None:
        # Convert history to JSON-serializable format
        serializable_history = {phase: {metric: [float(value) for value in values] for metric, values in phase_history.items()} for phase, phase_history in history.items()}
        
        # Save training history
        history_path = RESULTS_DIR / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        
        # Plot and save training curves
        plot_training_curves(history)
    
    return metrics

def plot_training_curves(history):
    """
    Plot and save training curves
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['phase1']['accuracy'], label='Phase 1 Train')
    plt.plot(history['phase1']['val_accuracy'], label='Phase 1 Val')
    plt.plot(history['phase2']['accuracy'], label='Phase 2 Train')
    plt.plot(history['phase2']['val_accuracy'], label='Phase 2 Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['phase1']['loss'], label='Phase 1 Train')
    plt.plot(history['phase1']['val_loss'], label='Phase 1 Val')
    plt.plot(history['phase2']['loss'], label='Phase 2 Train')
    plt.plot(history['phase2']['val_loss'], label='Phase 2 Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'training_curves.png')
    plt.close()

if __name__ == '__main__':
    # Train the model
    train_df, val_df = load_data_aptos2019()
    train_generator, val_generator = create_data_generators(train_df, val_df)
    
    # Calculate class weights
    class_weights = get_class_weights(train_df)
    
    # Train the model
    model, history = train_model(train_generator, val_generator, class_weights)
    
    # Use validation generator for evaluation
    test_generator = val_generator
    
    # Evaluate the model
    metrics = evaluate_model(model, test_generator, history)
    
    # Print results
    print("\nModel Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 