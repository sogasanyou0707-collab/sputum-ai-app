import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
import base64
import io
from datetime import datetime

# --- 設定情報 ---
# ★ステップ2で取得した「ウェブアプリのURL」をここに貼ります
GAS_URL = 'https://script.google.com/macros/s/AKfycbw0HZ1S7bdVHe3sEkudhZfaZ8lc3rFJ0aFhB9iiXFA6M6CVST3ZKg7nsbFByeyPSHD_/exec'

# --- メインアプリ部分 ---
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = f.readlines()
    return model, labels

model, labels = load_model_and_labels()

st.title("喀痰AI：クラウド保存版（GAS連携）")

image_file = st.camera_input("撮影してGoogleドライブへ送信")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    
    # AI判定
    size = (224, 224)
    resized_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(resized_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index].strip()[2:]
    confidence = prediction[0][index] * 100

    st.success(f"AI判定: {class_name} ({confidence:.1f}%)")

    # Googleドライブへアップロード（GAS経由）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name}_{timestamp}.jpg"
    
    with st.spinner('Googleドライブへ保存中...'):
        try:
            # 携帯から送るために画像を少し軽量化（通信エラー防止）
            image.thumbnail((800, 800))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=80)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            file_data = f"data:image/jpeg;base64,{img_str}"
            
            # GASへ送信
            response = requests.post(GAS_URL, data={
                "filename": filename,
                "fileData": file_data
            })
            
            if "保存成功" in response.text:
                st.info(f"✅ Googleドライブに保存完了！")
            else:
                st.error(f"保存エラー: {response.text}")
                
        except Exception as e:
            st.error(f"通信エラー: {e}")
