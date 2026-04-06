import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
import base64
import io
from datetime import datetime

# --- 設定情報 ---
# あなたのGASウェブアプリのURLをここに貼ってください
GAS_URL = 'https://script.google.com/macros/s/AKfycbxobhwhk6IDwStPj3NB1J7-ufRolBoP4t6Mc8KfjHD-75A4hPzes0EA9kW-q4FwVV4/exec'

# --- モデル読み込み ---
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = f.readlines()
    return model, labels

model, labels = load_model_and_labels()

st.title("喀痰AI：高画質・フォルダ自動振分版")

# --- 入力方法の選択 ---
input_method = st.radio("入力方法を選択", ("カメラで直接撮影", "高画質写真を選択（推奨）"))

if input_method == "カメラで直接撮影":
    image_file = st.camera_input("撮影")
else:
    # 複数枚一気に送ることも想定できますが、まずは1枚ずつ確実に
    image_file = st.file_uploader("アルバムから選択またはカメラアプリで撮影", type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    
    # プレビュー表示
    st.image(image, caption='選択された画像', use_column_width=True)
    
    # --- AI判定処理 ---
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

    st.success(f"AI判定結果: {class_name} (確信度: {confidence:.1f}%)")

    # --- Googleドライブへ保存（高画質設定） ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # GAS側でフォルダ分けするために、ファイル名の先頭に判定結果を付ける
    filename = f"{class_name}_{timestamp}.jpg"
    
    if st.button("Googleドライブへ保存実行"):
        with st.spinner('最高画質で保存中...'):
            try:
                # 軽量化処理(thumbnail)を削除し、品質(quality)を最大級にアップ
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=95) 
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                file_data = f"data:image/jpeg;base64,{img_str}"
                
                # GASへ送信
                response = requests.post(GAS_URL, data={
                    "filename": filename,
                    "fileData": file_data
                })
                
                if "保存成功" in response.text:
                    st.info(f"✅ {class_name} フォルダに保存完了しました！")
                else:
                    st.error(f"保存エラー: {response.text}")
                    
            except Exception as e:
                st.error(f"通信エラー: {e}")
