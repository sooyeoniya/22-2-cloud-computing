from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from keras.models import load_model
import streamlit as st
st.title("클라우드컴퓨팅과AI서비스")
st.write("##### 2020136129 최수연")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_Model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r', encoding='UTF8').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open(
    "C:/Users/choi6/Desktop/3-2/클라우드컴퓨팅과AI서비스/2020136129최수연_사진데이터/test2.jpg").convert('RGB')
st.image(image)
# resize the image to a 224x224 with the same strategy as in TM2:
# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

st.write('Pose:', class_name, end='')
st.write('Confidence score:', confidence_score)
if '0' in class_name:
    st.write('Facial Expression: 슬픈 표정')
    st.write('Color: #d3d3d3, #0067a3')
elif '1' in class_name:
    st.write('Facial Expression: 오열하는 표정')
    st.write('Color: #3F3D3C, #000080')
