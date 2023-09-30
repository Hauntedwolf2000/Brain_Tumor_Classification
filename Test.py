import os
import numpy as np
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the trained model
model = load_model("brain_tumor_classification_model.h5")

# Define the image size
img_size = (220, 220)
new_image_path = 'D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\demo\\gg (18).jpg'  # Updated file path

# Load and preprocess the new image using EfficientNet's preprocessing
new_img = image.load_img(new_image_path, target_size=img_size)
new_img = image.img_to_array(new_img)
new_img = np.expand_dims(new_img, axis=0)
new_img = preprocess_input(new_img)  # No need to specify 'mode'

# Use the trained model to make predictions
predictions = model.predict(new_img)

# Interpret the model's predictions
predicted_class = np.argmax(predictions)
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Define your class labels
predicted_label = class_labels[predicted_class]

print(f"Predicted Tumor Type: {predicted_label}")

img = mpimg.imread(new_image_path)
plt.imshow(img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()






