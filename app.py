import numpy as np
import streamlit as st
from PIL import Image
import uuid
import os
import cv2
from transformers import pipeline

st.set_page_config(
    page_title="Image Captioning & Editing",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("Image Editing & Captioning")
    # initialize each of variables (id,text and image)
    if "id" not in st.session_state:
        st.session_state.id = None
    if "text" not in st.session_state:
        st.session_state.text = None
    if "image" not in st.session_state:
        st.session_state.image = None
    # upload image file
    image_file = st.file_uploader("Upload image",
                                  type=['jpeg',
                                        'jpg',
                                        'png',
                                        'jfif'])
    if image_file:
        # read image file
        st.session_state.image = Image.open(image_file)
    if st.session_state.image:
        st.image(st.session_state.image)
        actions = st.selectbox("Choose Which Process you want it:",
                               (None, "Caption", "Edit"))
        if actions == "Caption":
            # get image to caption model
            pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            with st.spinner("Processing..."):
                st.session_state.id = str(uuid.uuid4())
                st.header("Image Captioning")
                st.session_state.text = pipe(st.session_state.image)[0]["generated_text"]
                with open(f"caption_{st.session_state.id}.txt", "w") as f:
                    f.write(st.session_state.text)
        file_path = f"caption_{st.session_state.id}.txt"
        if os.path.exists(file_path):
            # show caption text and allow user to edit it
            st.text_area("Captioned text:",
                         value=st.session_state.text)
            # allow user to download caption
            st.download_button("download caption",
                               f"caption_{st.session_state.id}.txt")
        if actions == "Edit":
            st.header("Image Editing")
            st.subheader("Image Resize")
            # adjust new width for image
            image_width = st.number_input("Image Width",
                                          value=st.session_state.image.width)
            # adjust new height for image
            image_height = st.number_input("Image Height",
                                           value=st.session_state.image.height)
            st.subheader("Image Rotate")
            # adjust rotate degree of image
            rotate = st.number_input("Image Rotate",
                                     min_value=0,
                                     max_value=360)
            st.subheader("Image Filter")
            # option all filters
            filter_options = (
                None, "Blur", "GaussianBlur", "MedianBlur", "BilateralFilter", "Sobel_x", "Sobel_y", "Scharr_x",
                "Scharr_y", "Canny", "Threshold", "AdaptiveThreshold", "Erode", "Dilate", "MorphologyEx")
            filter_ = st.selectbox("Choose Filters",
                                   options=filter_options)
            proc = st.button("process")
            if proc:
                processed_image = st.session_state.image.resize((image_width, image_height)).rotate(rotate)

                if filter_ is not None:
                    array_image = np.array(processed_image)
                    if filter_ == "Blur":
                        array_image = cv2.blur(array_image, (3, 3))
                    elif filter_ == "GaussianBlur":
                        array_image = cv2.GaussianBlur(array_image, (5, 5), 0)
                    elif filter_ == "MedianBlur":
                        array_image = cv2.medianBlur(array_image, 5)
                    elif filter_ == "BilateralFilter":
                        array_image = cv2.bilateralFilter(array_image, 9, 75, 75)
                    elif filter_ == "Sobel_x":
                        array_image = cv2.Sobel(array_image, cv2.CV_64F, 1, 0, ksize=3)
                    elif filter_ == "Sobel_y":
                        array_image = cv2.Sobel(array_image, cv2.CV_64F, 0, 1, ksize=3)
                    elif filter_ == "Scharr_x":
                        array_image = cv2.Scharr(array_image, cv2.CV_64F, 1, 0)
                    elif filter_ == "Scharr_y":
                        array_image = cv2.Scharr(array_image, cv2.CV_64F, 0, 1)
                    elif filter_ == "Canny":
                        array_image = cv2.Canny(array_image, 100, 200)
                    elif filter_ == "Threshold":
                        _, array_image = cv2.threshold(array_image, 127, 255, cv2.THRESH_BINARY)
                    elif filter_ == "AdaptiveThreshold":
                        # Convert to grayscale
                        gray_image = cv2.cvtColor(array_image, cv2.COLOR_RGB2GRAY)
                        array_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                            cv2.THRESH_BINARY, 11, 2)
                    elif filter_ == "Erode":
                        array_image = cv2.erode(array_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    elif filter_ == "Dilate":
                        array_image = cv2.dilate(array_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    elif filter_ == "MorphologyEx":
                        array_image = cv2.morphologyEx(array_image, cv2.MORPH_CLOSE,
                                                       cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

                    processed_image = Image.fromarray(array_image.astype(np.uint8))

                st.session_state.image = processed_image
                st.image(st.session_state.image)

    else:
        # delete caption file if user change image
        file_path = f"caption_{st.session_state.id}.txt"
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    main()
