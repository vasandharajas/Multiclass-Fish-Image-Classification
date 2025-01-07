import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("C:/Users/anand/Desktop/Image ClassFish Project/best_fish_model.keras")

# Define class names (replace with your actual class names)
class_names = ['animal_fish', 'animal_fish_bass', 'fish_sea_food_black_sea_sprat',
               'fish_sea_food_gilt_head_bream', 'fish_sea_food_hourse_mackerel',
               'fish_sea_food_red_mullet', 'fish_sea_food_red_sea_bream',
               'fish_sea_food_sea_bass', 'fish_sea_food_shrimp',
               'fish_sea_food_striped_red_mullet', 'fish_sea_food_trout']

st.title("Fish Classification App")
st.write("Upload an image of a fish, and the model will predict its category along with confidence scores.")

# Function to predict the fish category
def predict(image):
    img = image.resize((224, 224))  # Resize image to match input size expected by the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model's input shape
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return predicted_class, confidence, scores

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence, scores = predict(image)
    st.write(f"**Predicted Category: {predicted_class}**")
    st.write(f"**Confidence: {confidence:.2f}%**")
    
    # Display confidence scores for all classes
    st.write("**Confidence Scores for all categories:**")
    for i, score in enumerate(scores):
        st.write(f"{class_names[i]}: {score:.2%}")