import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import json
import h5py
import os

st.set_page_config(page_title="Classificador de Números", layout="centered")
st.title("Classificador de Números (1-9)")
st.markdown("Puja una imatge d'un número escrit a mà i la IA intentarà reconèixer-lo.")

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def conv2d(x, filters, biases):
    h, w, c = x.shape
    fh, fw, _, nf = filters.shape
    out = np.zeros((h - fh + 1, w - fw + 1, nf))
    for f in range(nf):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j, f] = np.sum(x[i:i+fh, j:j+fw, :] * filters[:,:,:,f]) + biases[f]
    return relu(out)

def maxpool(x):
    h, w, c = x.shape
    out = np.zeros((h//2, w//2, c))
    for i in range(h//2):
        for j in range(w//2):
            out[i, j, :] = np.max(x[i*2:i*2+2, j*2:j*2+2, :], axis=(0,1))
    return out

def load_weights():
    weights = {}
    with h5py.File("model_numeros.weights.h5", "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        f.visititems(collect)
    return weights

def predict(img_array, weights):
    keys = sorted(weights.keys())
    w_list = [weights[k] for k in keys]
    
    x = img_array[0]
    x = conv2d(x, w_list[0], w_list[1])
    x = maxpool(x)
    x = conv2d(x, w_list[2], w_list[3])
    x = maxpool(x)
    x = x.flatten()
    x = relu(x @ w_list[4] + w_list[5])
    x = softmax(x @ w_list[6] + w_list[7])
    return x

uploaded_file = st.file_uploader("Puja una imatge en format JPG o PNG", type=["jpg", "jpeg", "png"])

if not os.path.exists("model_numeros.weights.h5"):
    st.error("El model no s'ha trobat.")
elif uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Imatge pujada", use_container_width=True)

        img_array = np.array(image.resize((28, 28))) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        with st.spinner("Analitzant..."):
            weights = load_weights()
            probs = predict(img_array, weights)

        numero_predicho = np.argmax(probs) + 1
        confianza = float(probs[np.argmax(probs)]) * 100

        st.success(f"La IA creu que és el número **{numero_predicho}** amb un {confianza:.2f}% de confiança.")
        st.info("Recorda: el model funciona millor amb números escrits a mà sobre fons blanc.")

    except UnidentifiedImageError:
        st.error("No s'ha pogut llegir la imatge. Puja un fitxer JPG o PNG vàlid.")