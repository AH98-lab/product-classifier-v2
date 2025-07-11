import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

MODEL_URL = "https://drive.google.com/uc?export=download&id=1eZTh4-3E78ZQnINE0DX0p3ErNwXvnhOM"
MODEL_PATH = "keras_model.h5"

# تحميل النموذج من Google Drive إذا لم يكن موجودًا
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 يتم تحميل نموذج الذكاء الاصطناعي..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ تم تحميل النموذج بنجاح!")

# إعداد الواجهة
st.set_page_config(page_title="تصنيف المنتجات بالذكاء الاصطناعي", layout="centered")
st.title("🧠 تصنيف صور المنتجات")
st.markdown("ارفع صورة المنتج وسيتعرف عليها الذكاء الاصطناعي تلقائيًا")

# رفع الصورة
uploaded_file = st.file_uploader("📤 قم برفع صورة المنتج", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="✅ الصورة التي تم رفعها", use_column_width=True)

    # تحميل النموذج
    model = tf.keras.models.load_model(MODEL_PATH)

    # معالجة الصورة
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.asarray(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_names = ["ملابس", "أثاث", "أدوات منزلية", "إلكترونيات", "أغذية"]

    st.subheader("🔍 النتيجة:")
    st.success(f"تم التعرف على: **{class_names[np.argmax(prediction)]}**")
