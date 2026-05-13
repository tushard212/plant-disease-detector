import cv2
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras as efn
import streamlit as st

st.set_page_config(page_title='Virtue', page_icon = 'assets/images/logo.png')

# Title and Description
st.title("Virtue Image")
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not") 

model = tf.keras.models.load_model('model.h5')

uploaded_file = st.file_uploader('Choose your image', type=['png', 'jpg'])

predictions_map = {0:'is healthy', 1:'has Multiple Diseases', 2:'has rust(Fungus)', 3:'has scab(Bacterial)'}

predictions_sol_rust = {0:'Choose resistant varieties', 1:'Keep leaves dry', 2:'Clean up debris', 3:'Use fungicides'}

predictions_sol_scap ={0:'Choose resistant varieties' , 1 : "Maintain good sanitation" , 2:"Water at the base" , 3 : "Use copper-based fungicides"
}

predictions_sol_vast = {0 : 'Choose disease-resistant varieties' , 1 : 'Practice good sanitation' , 2 : 'Water appropriately' , 3 : 'Rotate crops: Planting different crops in different areas of your garden each year can help prevent the buildup of soil-borne diseases.'}

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, use_column_width=True)
    
    resized_image = np.array(image.resize((512,512)))/255. # Resize image and divide pixel number by 255. for having values between 0 and 1 (normalize it)
    
    image_batch = resized_image[np.newaxis, :, :, :]
    
    predictions_arr = model.predict(image_batch)
    
    predictions = np.argmax(predictions_arr)

    result_text = f'The plant leaf {predictions_map[predictions]} with {int(predictions_arr[0][predictions]*100)}% Uncovered area'

    if predictions == 0:
        st.success(result_text)
        st.text("No treatment required")
    elif predictions == 1:
        st.error(result_text)
        st.markdown("<p style='color: red;'>You can apply the following solution:</p>", unsafe_allow_html=True)
        st.text(predictions_sol_vast[0])
        st.text(predictions_sol_vast[1])
        st.text(predictions_sol_vast[2])
        st.text(predictions_sol_vast[3])
        # Convert the image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to segment the leaf from the background
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and draw a rectangle around the defected portion of the leaf
        for cnt in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Check if the contour is too small or too large
            if w < 10 or h < 10 or w > 100 or h > 100:
                continue

            # Draw a rectangle around the contour
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Show the image
        st.image(cv_image, use_column_width=True)   
        

    elif predictions == 2:
        st.error(result_text)
        st.markdown("<p style='color: red;'>You can apply the following solution:</p>", unsafe_allow_html=True)
        st.text(predictions_sol_rust[0])
        st.text(predictions_sol_rust[1])
        st.text(predictions_sol_rust[2])
        st.text(predictions_sol_rust[3])
        # Convert the image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to segment the leaf from the background
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and draw a rectangle around the defected portion of the leaf
        for cnt in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Check if the contour is too small or too large
            if w < 10 or h < 10 or w > 100 or h > 100:
                continue

            # Draw a rectangle around the contour
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Show the image
        st.image(cv_image, use_column_width=True)   
        

    else:
        st.error(result_text)
        st.markdown("<p style='color: red;'>You can apply the following solution:</p>", unsafe_allow_html=True)
        st.text(predictions_sol_scap[0])
        st.text(predictions_sol_scap[1])
        st.text(predictions_sol_scap[2])
        st.text(predictions_sol_scap[3])
        # Convert the image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to segment the leaf from the background
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and draw a rectangle around the defected portion of the leaf
        for cnt in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Check if the contour is too small or too large
            if w < 10 or h < 10 or w > 100 or h > 100:
                continue

            # Draw a rectangle around the contour
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Show the image
        st.image(cv_image, use_column_width=True)   
        
