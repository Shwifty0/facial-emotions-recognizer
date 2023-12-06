import io, time
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from model import resnetfc, MODEL, DEVICE

# Set tile for the app:
st.title(":pager: Facial Emotions Recognition System:")

# function for uploading image and applying transformations to it for predictions:
def input_image():
    image_file = st.file_uploader("**Upload an image expressing an Emotion:**", type=["jpg"])
    if image_file is not None:
        # Read the uploaded image as bytes
        image_bytes = image_file.read()
        
        # Convert the bytes to a PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Apply transformations to prepare the image for the model
        transform = T.Compose([
            T.Resize((224, 224)),  # Resize to 224x224 pixels
            T.ToTensor()  # Convert to PyTorch tensor
        ])
        transformed_img = transform(img).unsqueeze(0)  # Add a batch dimension
        

        st.image(img, caption="Uploaded Image", width= 500)
        # Return the transformed image tensor
        return transformed_img


transformed_image = input_image()

# Assign the new fc layer to the intitialized model
MODEL.fc = resnetfc()
# Load the saved "State Dictionary" of the trained model in Jupyter Notebook
MODEL.load_state_dict(torch.load("FacialEmo.pth", map_location= DEVICE))

# Make Predictions
class_names = ['Angry','Contempt','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
predict = st.button("Predict")
if predict:
    with st.spinner("Predicting..."):
        time.sleep(1)
        MODEL.eval()
        with torch.inference_mode():
            # Feed the uploaded image to the model
            y_pred_label = torch.argmax(torch.softmax(MODEL(transformed_image), dim = 1), dim = 1)
        # Show predictions
        st.success(f"This person seems: {class_names[y_pred_label]}", icon = "🤖")

# Markdowns for what the app is about
# Post a link to the Kaggle dataset
url = "https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition"
st.write(f":fire: Check out the dataset over here: {url}")

# Information about your project
st.info(":joystick: The dataset has only 152 samples of facial emotions with 8 different classes")
st.info(f":joystick: class_names = {class_names}")
st.info(":joystick: At the initial stages I trained the model with regular transformations and the model was overfitting")
st.info(":joystick: In order to solve the problem of overfitting, I used PyTorch's Data Augmentation tools in order to make it hard for the model to learn new patterns and not overfit")
st.info(":joystick: The model does not perform well with new data (Its trained on very specific type of facial features.)")

# Profile Pic and relevant links (LinkedIn, GitHub)
with st.sidebar:
    profile_pic = Image.open("Profile Pic.jpg") 
    linked_in_img = Image.open("linked_in.png")
    st.image(profile_pic, caption="Connect with me on LinkedIn and Drop a follow on my GitHub! Thanks", width= 250)
    st.write(":smiley:")
    st.markdown("https://github.com/Shwifty0")
    st.markdown("https://www.linkedin.com/in/muhammad-ozair-b12682177/")
    #st.markdown(linked_in_img)
    st.info("Hey there! My name is Muhammad Ozair, I am a Computer Engineer from Bahria University Karachi Campus. \n I am currently teaching myself to be a Machine Learning Engineer :sleuth_or_spy: \n This app is a part of my learning process. :smile:")
