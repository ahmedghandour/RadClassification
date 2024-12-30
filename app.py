import os
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle

# Helper functions
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize((64, 64))  # Resize to a fixed size
                    images.append(np.array(img).flatten())
                    labels.append(label)
                except Exception as e:
                    st.warning(f"Could not load image {img_path}: {e}")
    return np.array(images), np.array(labels)

def train_model(data_folder):
    st.write("Training model...")
    images, labels = load_images_from_folder(data_folder)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    st.success("Model trained and saved successfully!")
    st.write(f"Training accuracy: {model.score(X_train, y_train):.2f}")
    st.write(f"Test accuracy: {model.score(X_test, y_test):.2f}")

def classify_image(image_path):
    if not os.path.exists('model.pkl'):
        st.error("Model not found. Please train the model first.")
        return

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((64, 64))
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(img_array)
    return prediction[0]

# Streamlit app
st.title("Image Classification App")

data_folder = "data"

# Button to train the model
if st.button("Train Model"):
    if os.path.exists(data_folder):
        train_model(data_folder)
    else:
        st.error(f"Data folder '{data_folder}' does not exist.")

# File uploader to load and classify an image
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    prediction = classify_image("temp_image.jpg")
    if prediction:
        st.success(f"The image was classified as: {prediction}")
