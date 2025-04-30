import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
from tensorflow.keras.applications.densenet import preprocess_input

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image."""
    
    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool the gradients across all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    """Save Grad-CAM visualization."""
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap
    jet = plt.cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    # Save the superimposed image
    superimposed_img.save(cam_path)

def visualize_gradcam(model, image_path, last_conv_layer_name):
    """Generate and save Grad-CAM visualization for a given image."""
    
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Use DenseNet121 preprocessing
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save visualization
    cam_path = VISUALIZATION_DIR / f"gradcam_{Path(image_path).stem}.png"
    save_gradcam(image_path, heatmap, cam_path)
    
    return cam_path

def get_last_conv_layer_name(model):
    """Get the name of the last convolutional layer in the model."""
    # For DenseNet121, the last convolutional layer is typically 'conv5_block16_concat'
    if MODEL_NAME == 'densenet121':
        # Try to find the specific DenseNet layer by name
        try:
            return model.get_layer('conv5_block16_concat').name
        except:
            pass  # Fall back to general approach if layer not found
    
    # General approach - find the last convolutional layer
    for layer in reversed(model.layers):
        # Check if it's a convolutional layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Check inside if it's a composite layer with nested layers
        if hasattr(layer, 'layers'):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return inner_layer.name
    
    return None 