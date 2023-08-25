import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):

  
    model = load_model(weights_file)


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = img

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)


    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

  
    data[0] = normalized_image_array

    
    prediction = model.predict(data)

    
    return prediction.tolist()[0]



def main():


    st.title("城判別")


    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') 

        classNo = np.argmax(prediction)          
        st.caption(f'判定結果：{classNo}番')      


        pred0 = round(prediction[0],3) * 100  # 確率(%)
        pred1 = round(prediction[1],3) * 100  # 確率(%)

        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「名古屋城」です！")
        else:
            st.subheader(f"これは{pred1}％の確率で「金沢城」です！")



if __name__ == "__main__":
    main()


