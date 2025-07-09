###################################################
# To get rid of warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
###################################################

import sys
import getopt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###################################################

# Top-k accuracy metrics
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

###################################################

# Load MobileNetV2 and modify for 7 classes
mobile = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = mobile.output  # Use the output of the base MobileNetV2 model
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Pooling layer
x = Dropout(0.25)(x)  # Add a Dropout layer
x = BatchNormalization()(x)  # Add Batch Normalization layer
predictions = Dense(7, activation='softmax')(x)  # Updated for 7 classes
model = Model(inputs=mobile.input, outputs=predictions)

# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-23:]:
    layer.trainable = True

# Compile the model with a smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Use a smaller learning rate for fine-tuning
    loss='categorical_crossentropy', 
    metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy]
)

# Load the weights excluding the final layer mismatch
model.load_weights('model.h5', by_name=True, skip_mismatch=True)

###################################################

# Update class labels
class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

####################################################

# Handle command-line arguments
inputfile = 'ISIC_0024308.jpg'
opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile="])
for opt, arg in opts:
    if opt == '-h':
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg

####################################################

# Function to load and predict images
def loadImages(path):
    img = image.load_img(path, target_size=(224, 224))  # Resize image to match MobileNetV2 input
    img_data = image.img_to_array(img)  # Convert image to array
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
    img_data = tf.keras.applications.mobilenet_v2.preprocess_input(img_data)  # Preprocess for MobileNetV2
    features = np.array(model.predict(img_data))  # Predict the class
    y_classes = features.argmax(axis=-1)  # Get the index of the highest probability
    accuracy = features[0][y_classes[0]]  # Confidence of the predicted class
    print(f"Confidence (Accuracy) of the prediction: {accuracy * 100:.2f}%")
    return y_classes

###################################################

# Predict and display the class label
x = loadImages(inputfile)
print("Predicted output is:", class_labels[x[0]])

