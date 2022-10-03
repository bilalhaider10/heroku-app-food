import tensorflow as tf
model = tf.keras.models.load_model('my_model5.hdf5')
import streamlit as st
st.write("""
         # Food21 image classification and Prediction
         """
         )
st.write("This is a simple image classification web app to predict food")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_resize = image/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a apple_pie!")
    elif np.argmax(prediction) == 1:
        st.write("It is a baby_black_ribs!")
    elif np.argmax(prediction) == 2:
        st.write("It is a baklava!")
    elif np.argmax(prediction) == 3:
        st.write("It is a beef_carpaccio!")
    elif np.argmax(prediction) == 4:
        st.write("It is a beef_tartare!")
    elif np.argmax(prediction) == 5:
        st.write("It is a beet_salad!")
    elif np.argmax(prediction) == 6:
        st.write("It is a beignets!")
    elif np.argmax(prediction) == 7:
        st.write("It is a bibimbap!")
    elif np.argmax(prediction) == 8:
        st.write("It is a bread_pudding!")
    elif np.argmax(prediction) == 9:
        st.write("It is a breakfast_burrito!")
    elif np.argmax(prediction) == 10:
        st.write("It is a bruschetta!")
    elif np.argmax(prediction) == 11:
        st.write("It is a caesar_salad!")
    elif np.argmax(prediction) == 12:
        st.write("It is a cannoli!")
    elif np.argmax(prediction) == 13:
        st.write("It is a caprese_salad!")
    elif np.argmax(prediction) == 14:
        st.write("It is a carrot_cake!")
    elif np.argmax(prediction) == 15:
        st.write("It is a ceviche!")
    elif np.argmax(prediction) == 16:
        st.write("It is a cheesecake!")
    elif np.argmax(prediction) == 17:
        st.write("It is a cheese_plate!")
    elif np.argmax(prediction) == 18:
        st.write("It is a chicken_curry!")
    elif np.argmax(prediction) == 19:
        st.write("It is a chicken_quesadilla!")
    else:
        st.write("big class")
    
    st.text("Probability (0: 'apple_pie', 1: 'baby_back_ribs', 2:'baklava', 3:'beef_carpaccio',4:'beef_tartare', 5: 'beet_salad', 6:'beignets', 7:'bibimbap',8: 'bread_pudding', 9:'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake','cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'big_class' ")
    st.write(prediction)