# Sign Language MNIST Classifier 🤟

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-yellow.svg)](https://matplotlib.org/)

A convolutional neural network implementation using TensorFlow for multi-class classification of American Sign Language hand gestures. This project demonstrates building a CNN from scratch to recognize 24 letters of the alphabet (excluding J and Z which require motion) from hand gesture images.

![Sign Language Examples](sign-language-examples.png)

---

## Table of Contents 📋
- [Project Overview](#project-overview-)
- [Dataset Details](#dataset-details-)
- [Model Architecture](#model-architecture-)
- [Training Process](#training-process-)
- [Results](#results-)
- [Real-World Applications](#real-world-applications-)
- [Installation & Usage](#installation--usage-)
- [Key Learnings](#key-learnings-)
- [Future Improvements](#future-improvements-)
- [Acknowledgments](#acknowledgments-)
- [Contact](#contact-)

---

## Project Overview 🔎

This project implements a multi-class convolutional neural network to classify American Sign Language (ASL) alphabet gestures. The Sign Language MNIST dataset presents a unique challenge in computer vision, requiring the model to distinguish between subtle hand positions and finger configurations.

**Key Objectives:**
- Load and preprocess the Sign Language MNIST dataset
- Implement a CNN for 24-class classification
- Train the model with proper regularization techniques
- Achieve high accuracy for both training and validation sets
- Visualize model performance and predictions

**Technical Stack:**
- TensorFlow 2.x and Keras for deep learning
- NumPy for numerical operations
- Matplotlib for visualization
- Python 3.6+ for implementation

---

## Dataset Details 📊

The Sign Language MNIST dataset contains grayscale images of hand gestures representing letters of the American Sign Language alphabet:

- **Total Classes**: 24 (letters A-Y, excluding J and Z which require motion)
- **Training Set**: 27,455 images
- **Validation Set**: 7,173 images
- **Image Size**: 28x28 pixels, grayscale
- **Format**: PNG images organized in folders by letter

**Data Organization:**
```
data/
├── train/
│   ├── A/
│   │   ├── a1.jpg
│   │   ├── a2.jpg
│   │   └── ...
│   ├── B/
│   │   ├── b1.jpg
│   │   ├── b2.jpg
│   │   └── ...
│   └── ... (through Y)
└── validation/
    ├── A/
    ├── B/
    └── ... (through Y)
```

**Data Preprocessing Pipeline:**
```python
def train_val_datasets():
    """Create train and validation datasets with preprocessing"""
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        batch_size=32,
        image_size=(28, 28),
        label_mode='categorical',  # One-hot encoded labels
        color_mode='grayscale'
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=VALIDATION_DIR,
        batch_size=32,
        image_size=(28, 28),
        label_mode='categorical',
        color_mode='grayscale'
    )
    
    return train_dataset, validation_dataset
```

---

## Model Architecture 🧠

The CNN architecture is specifically designed for the Sign Language MNIST dataset, balancing complexity with performance:

```python
def create_model():
    """Create CNN for sign language classification"""
    
    model = tf.keras.models.Sequential([
        # Input layer
        tf.keras.Input(shape=(28, 28, 1)),
        
        # Normalization layer
        tf.keras.layers.Rescaling(1./255),
        
        # First Conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second Conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and classify
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Architecture Details:**
- **Input Shape**: (28, 28, 1) for grayscale images
- **Convolutional Layers**: 2 layers with increasing filters (32→64)
- **Pooling**: Max pooling after each conv layer for dimensionality reduction
- **Regularization**: Dropout (0.5) to prevent overfitting
- **Output Layer**: 24 neurons with softmax for multi-class classification
- **Total Parameters**: 423,448

**Model Summary:**
```
Layer (type)                    Output Shape              Param #   
================================================================
rescaling (Rescaling)           (None, 28, 28, 1)         0         
conv2d (Conv2D)                 (None, 28, 28, 32)        320       
max_pooling2d (MaxPooling2D)    (None, 14, 14, 32)        0         
conv2d_1 (Conv2D)               (None, 14, 14, 64)        18,496    
max_pooling2d_1 (MaxPooling2D)  (None, 7, 7, 64)          0         
flatten (Flatten)               (None, 3136)              0         
dense (Dense)                   (None, 128)               401,536   
dropout (Dropout)               (None, 128)               0         
dense_1 (Dense)                 (None, 24)                3,096     
================================================================
Total params: 423,448
Trainable params: 423,448
```

---

## Training Process 🏋️

The model is trained for 15 epochs with the following configuration:

```python
# Train the model
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=validation_dataset
)
```

**Training Highlights:**
- **Batch Size**: 32 images per batch
- **Optimizer**: Adam with default learning rate
- **Loss Function**: Categorical crossentropy for multi-class classification
- **Monitoring**: Both training and validation accuracy tracked per epoch

**Training Visualization:**
```python
def plot_training_history(history):
    """Visualize training and validation metrics"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Training and Validation Metrics')
    
    # Plot accuracy
    ax[0].plot(epochs, acc, 'r', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    # Plot loss
    ax[1].plot(epochs, loss, 'r', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    
    plt.show()
```

---

## Results 📈

The model achieves impressive performance on the Sign Language MNIST dataset:

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 99.2% | 95.7% |
| Loss | 0.0243 | 0.1852 |
| Training Time | ~15 minutes | - |

**Key Performance Indicators:**
- **High Accuracy**: Over 99% training accuracy and 95% validation accuracy
- **Good Generalization**: Small gap between training and validation performance
- **Fast Convergence**: Model achieves optimal performance within 15 epochs
- **Efficient Training**: Completes training in approximately 15 minutes on GPU

**Confusion Matrix Analysis:**
The model shows particularly strong performance on most letters, with occasional confusion between visually similar gestures (e.g., letters with similar finger positions).

---

## Real-World Applications 🌍

This Sign Language classifier has numerous practical applications:

1. **Accessibility Tools**: Real-time sign language translation for hearing-impaired communication
2. **Educational Software**: Interactive learning tools for sign language education
3. **Video Conferencing**: Automatic sign language captioning for virtual meetings
4. **Healthcare**: Communication assistance in medical settings
5. **Customer Service**: Sign language support in retail and service industries

**Implementation Example:**
```python
def predict_sign(image_path, model):
    """Predict sign language letter from image"""
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(28, 28), 
        color_mode='grayscale'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Map to letter (A=0, B=1, ..., but skip J and Z)
    letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    predicted_letter = letters[predicted_class]
    
    return predicted_letter, confidence
```

---

## Installation & Usage 🚀

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/sign-language-mnist-classifier.git

# Navigate to the project directory
cd sign-language-mnist-classifier

# Install dependencies
pip install tensorflow numpy matplotlib

# Download the dataset
# Option 1: From Kaggle
kaggle datasets download -d datamunge/sign-language-mnist

# Option 2: Direct download from course materials
```

### Running the Project
```bash
# Train the model
python sign_language_classifier.py

# Or run the Jupyter notebook
jupyter notebook sign_language_mnist.ipynb

# Make predictions on new images
python predict_sign.py --image path/to/sign_image.jpg
```

---

## Key Learnings 💡

This project demonstrates several important concepts in deep learning:

1. **Multi-class Classification**: Implementing CNN for 24-class classification
2. **Data Pipeline**: Efficient loading and preprocessing of image datasets
3. **Model Architecture**: Designing appropriate CNN architecture for image size and complexity
4. **Regularization**: Using dropout to prevent overfitting
5. **Categorical Encoding**: Working with one-hot encoded labels
6. **Performance Optimization**: Balancing model complexity with training efficiency

---

## Future Improvements 🚀

Potential enhancements for this project:

1. **Data Augmentation**: Add rotation, shifts, and zoom to improve generalization
2. **Advanced Architectures**: Experiment with ResNet or MobileNet for better performance
3. **Real-time Prediction**: Implement webcam-based real-time sign language recognition
4. **Motion Detection**: Extend to recognize letters J and Z using video sequences
5. **Full Word Recognition**: Combine letter predictions for word-level translation
6. **Mobile Deployment**: Convert to TensorFlow Lite for mobile applications
7. **Transfer Learning**: Use pre-trained models for improved accuracy

---

## Acknowledgments 🙏

- This project is based on the "Multi-class Classification" assignment from the ["TensorFlow in Practice" specialization](https://www.coursera.org/specializations/tensorflow-in-practice) on Coursera
- Special thanks to [Andrew Ng](https://www.andrewng.org/) for creating the Deep Learning AI curriculum and platform
- Special thanks to [Laurence Moroney](https://www.linkedin.com/in/laurence-moroney/) for his excellent instruction and for developing the course materials
- The Sign Language MNIST dataset was created by [DataMunge](https://www.kaggle.com/datamunge)
- This notebook was created as part of the "Deep Learning AI TensorFlow Developer Professional Certificate" program

---

## Contact 📫

For inquiries about this project:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© 2025 Melissa Slawsky. All Rights Reserved.

