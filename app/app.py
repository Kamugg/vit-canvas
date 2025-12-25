from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import onnxruntime

@st.cache_resource
def get_onnx_session(pth: Path) -> onnxruntime.InferenceSession:
    """
    Loads ONNX ViT.
    :return: ONNX session
    """
    ort_session = onnxruntime.InferenceSession(pth,
                                               providers=['CPUExecutionProvider'])
    return ort_session

def preprocess_image(x: np.ndarray) -> np.ndarray:
    """
    Preprocesses the drawn digit to feed it to the ViT.
    :param x: The drawn digit.
    :return: Preprocessed drawn digit.
    """

    x = (x / 255.0).astype(np.float32)
    x = (x - 0.1307) / 0.3081
    x = x[np.newaxis, np.newaxis, ...]
    return x

def forward_pass(session: onnxruntime.InferenceSession, prep_img: np.ndarray) -> dict:
    """
    Performs a forward pass of the ViT model and returns both the non-softmaxed scores and the attention matrices.
    :param session: ONNX session.
    :param prep_img: The preprocessed drawn digit.
    :return: Dictionary of non-softmaxed scores and attention matrices.
    """

    ort_inputs = {session.get_inputs()[0].name: prep_img}
    ort_outs = session.run(None, ort_inputs)
    output_names = [output.name for output in session.get_outputs()]
    results_dict = dict(zip(output_names, ort_outs))
    return results_dict

def overlay_attention(digit: np.ndarray, att: np.ndarray) -> np.ndarray:
    """
    Overlays the attention matrix with the drawn digit.
    :param digit: The drawn (pixelated) digit.
    :param att: The attention matrix.
    :return: The overlayed drawn digit.
    """

    # Convertion to BGR
    digit_bgr = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)
    att = att[..., np.newaxis]

    # Green layer
    green_layer = np.zeros((CANVAS_BASE * CANVAS_FACTOR, CANVAS_BASE * CANVAS_FACTOR, 3),
                           dtype=np.float32)
    green_layer[:] = (0, 255, 0)
    digit_float = digit_bgr.astype(np.float32)

    # Alpha blending
    blended = (green_layer * att) + (digit_float * (1.0 - att))
    return blended.astype(np.uint8)

# Constants
CANVAS_FACTOR = 12
CANVAS_BASE = 28
STROKE_WIDTH = 15

# Model selection

models = {
    "vit-fast": "Lightweight, optimized for speed.",
    "vit-large": "Bigger, more accurate, but slower."
}
st.sidebar.header("Select model:")
selected_model = st.sidebar.radio(
    "select_model",
    options=models.keys(),
    format_func=lambda x: f"**{x}**",
    captions=models.values(),
    index=0,
    label_visibility="hidden",
)
if selected_model == "vit-fast":
    model_path = Path('./experiments/checkpoints/vit-fast/microvit_run_3.onnx')
elif selected_model == "vit-large":
    model_path = Path('./experiments/checkpoints/vit-large/microvit_run_34.onnx')
else:
    raise ValueError("Invalid model name.")

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Digit classifier with ViT</h1>", unsafe_allow_html=True)
st.space('medium')

col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment='top')

# Canvas column
with col1:
    st.write("### Draw a digit:")
    canvas_result = st_canvas(
        fill_color="rgb(0,0,0)",
        stroke_width=STROKE_WIDTH,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(0,0,0)",
        update_streamlit=True,
        height=int(CANVAS_BASE * CANVAS_FACTOR),
        width=int(CANVAS_BASE * CANVAS_FACTOR),
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas",
    )

# Showcase to the user the resized input image for the ViT
with col2:
    st.write("### What the ViT sees:")
    image = canvas_result.image_data
    if image is not None:

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        # "Pixelate". This is the vit raw input image
        mnist_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

        # Resize to show to the user
        display_image_pixelated = cv2.resize(
            mnist_image,
            (CANVAS_FACTOR * CANVAS_BASE, CANVAS_FACTOR * CANVAS_BASE),
            interpolation=cv2.INTER_NEAREST
        )
        st.image(display_image_pixelated)

# Forward pass is skipped if the canvas is empty
if (canvas_result.image_data is not None) and (np.sum(canvas_result.image_data[:, :, :3]) > 0):
    img = preprocess_image(mnist_image)
    sess = get_onnx_session(model_path)
    out = forward_pass(sess, img)
    scores = out['scores']
    expsum = np.sum(np.exp(scores))
    smaxed_scores = np.exp(scores) / expsum
else:
    smaxed_scores = np.zeros((1, 10))

# Column with the softmaxed scores
with col3:
    st.write("Probabilities:")
    with st.container(border=True):
        for i, s in enumerate(smaxed_scores[0]):
            row_col1, row_col2 = st.columns([1, 6])

            with row_col1:
                st.markdown(f"**Digit {i}**")

            with row_col2:
                st.progress(float(s))

# Showcase attention matrices, this pass is skipped if the canvas is empty
if (canvas_result.image_data is not None) and (np.sum(canvas_result.image_data[:, :, :3]) > 0):
    num_layers = len(out)-1
    num_heads = out['cls_att_tblock_0'].shape[1]
    for i in range(num_layers):
        st.write(f'### Transformer block {i+1}')
        with st.container(border=True):
            cols = st.columns(num_heads, vertical_alignment='center')
            for c, col in enumerate(cols):
                with col:
                    att_map = out[f'cls_att_tblock_{i}'][0, c]
                    resized_att_map = cv2.resize(att_map,
                                           (CANVAS_FACTOR * CANVAS_BASE, CANVAS_FACTOR * CANVAS_BASE),
                                                 interpolation=cv2.INTER_NEAREST)
                    st.image(overlay_attention(display_image_pixelated, resized_att_map),
                             caption=f'Head {c}',
                             width="stretch")
        st.space('small')