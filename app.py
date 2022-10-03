import tensorflow as tf
model = tf.keras.models.load_model('my_model5.hdf5')
import streamlit as st
st.write("""
         Food Classification
         """
         )
st.write("Image classification web app to predict the type food made on streamlit")
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
        st.write("Predicted: apple_pie!")
    elif np.argmax(prediction) == 1:
        st.write("Predicted: baby_black_ribs!")
    elif np.argmax(prediction) == 2:
        st.write("Predicted: baklava!")
    elif np.argmax(prediction) == 3:
        st.write("Predicted: beef_carpaccio!")
    elif np.argmax(prediction) == 4:
        st.write("Predicted: beef_tartare!")
    elif np.argmax(prediction) == 5:
        st.write("Predicted: beet_salad!")
    elif np.argmax(prediction) == 6:
        st.write("Predicted: beignets!")
    elif np.argmax(prediction) == 7:
        st.write("Predicted: bibimbap!")
    elif np.argmax(prediction) == 8:
        st.write("Predicted: bread_pudding!")
    elif np.argmax(prediction) == 9:
        st.write("Predicted: breakfast_burrito!")
    elif np.argmax(prediction) == 10:
        st.write("Predicted: bruschetta!")
    elif np.argmax(prediction) == 11:
        st.write("Predicted: caesar_salad!")
    elif np.argmax(prediction) == 12:
        st.write("Predicted: cannoli!")
    elif np.argmax(prediction) == 13:
        st.write("Predicted: caprese_salad!")
    elif np.argmax(prediction) == 14:
        st.write("Predicted: carrot_cake!")
    elif np.argmax(prediction) == 15:
        st.write("Predicted: ceviche!")
    elif np.argmax(prediction) == 16:
        st.write("Predicted: cheesecake!")
    elif np.argmax(prediction) == 17:
        st.write("Predicted: cheese_plate!")
    elif np.argmax(prediction) == 18:
        st.write("Predicted: chicken_curry!")
    elif np.argmax(prediction) == 19:
        st.write("Predicted: chicken_quesadilla!")
    else:
        st.write("Predicted: other categories")