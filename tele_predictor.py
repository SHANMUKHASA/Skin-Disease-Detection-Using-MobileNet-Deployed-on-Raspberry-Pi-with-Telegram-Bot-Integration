import os
import warnings
import numpy as np
import tensorflow as tf
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

# Replace with your bot token
# here i have a bot token Assigned which i cant reveal here

# Folder to save downloaded images
DOWNLOAD_FOLDER = "downloaded_images"

# Create the download folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Class labels for prediction
class_labels = ["bcc", "akiec"]

# Load MobileNetV2 and modify for 2 classes
mobile = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = mobile.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.25)(x)
x = BatchNormalization()(x)
predictions = Dense(len(class_labels), activation='softmax')(x)  # Updated for 2 classes
model = Model(inputs=mobile.input, outputs=predictions)

# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-23:]:
    layer.trainable = True

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=[categorical_accuracy, top_k_categorical_accuracy]
)

# Load the weights
model.load_weights('best_model.h5', by_name=True, skip_mismatch=True)

# Function to predict the class of an image
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_data = image.img_to_array(img)  # Convert to array
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
    img_data = tf.keras.applications.mobilenet_v2.preprocess_input(img_data)  # Preprocess for MobileNetV2
    predictions = model.predict(img_data)  # Predict the class probabilities
    predicted_class_index = np.argmax(predictions, axis=-1)  # Get the index of the highest probability
    confidence = np.max(predictions)  # Get the confidence of the prediction
    return class_labels[predicted_class_index[0]], confidence

# Handler for receiving images
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if the message contains a photo
    if update.message.photo:
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        # Get the file object
        file = await photo.get_file()
        # Define the file path to save the image
        file_path = os.path.join(DOWNLOAD_FOLDER, f"{file.file_id}.jpg")
        # Download the image
        await file.download_to_drive(file_path)
        # Notify the user
        await update.message.reply_text(f"Image saved as {file_path}")

        try:
            # Predict the class of the image
            predicted_class, confidence = predict_image_class(file_path)
            if confidence > 0.55:  # Check if confidence is greater than 70%
                await update.message.reply_text(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                await update.message.reply_text("Wrong image: Confidence is too low.")
        except Exception as e:
            await update.message.reply_text(f"Error processing image: {e}")
    else:
        await update.message.reply_text("Please send an image.")

# Main function to start the bot
def main():
    # Create the application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add a handler for images
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start the bot
    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
