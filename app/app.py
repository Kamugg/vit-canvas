import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import onnx
import onnxruntime

@st.cache_resource
def get_onnx_session():
    onnx_model = onnx.load('experiments/chkps/microvit_run_0.onnx')
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession('experiments/chkps/microvit_run_35.onnx',
                                               providers=['CPUExecutionProvider'])
    return ort_session

def preprocess_image(x: np.ndarray) -> np.ndarray:
    x = (x / 255.0).astype(np.float32)
    x = (x - 0.1307) / 0.3081
    x = x[None, None, ...]
    return x

def forward_pass(session: onnxruntime.InferenceSession, prep_img: np.ndarray) -> np.ndarray:
    ort_inputs = {session.get_inputs()[0].name: prep_img}
    ort_outs = session.run(None, ort_inputs)
    output_names = [output.name for output in session.get_outputs()]
    results_dict = dict(zip(output_names, ort_outs))
    scores = results_dict['scores']
    e_x = np.exp(scores)
    scores = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return scores

CANVAS_FACTOR = 12
CANVAS_BASE = 28
STROKE_WIDTH = 15

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Digit classifier with ViT</h1>", unsafe_allow_html=True)
st.markdown("<br>"*4, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment='top')

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

with col2:
    st.write("### What the ViT sees:")
    image = canvas_result.image_data
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        mnist_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        display_image_pixelated = cv2.resize(
            mnist_image,
            (CANVAS_FACTOR * CANVAS_BASE, CANVAS_FACTOR * CANVAS_BASE),
            interpolation=cv2.INTER_NEAREST
        )
        st.image(display_image_pixelated)

if (canvas_result.image_data is not None) and (np.sum(canvas_result.image_data[:, :, :3]) > 0):
    img = preprocess_image(mnist_image)
    sess = get_onnx_session()
    smaxed_scores = forward_pass(sess, img)
else:
    smaxed_scores = np.zeros((1, 10))

with col3:
    st.write("Probabilities:")
    with st.container(border=True):
        for i, s in enumerate(smaxed_scores[0]):
            row_col1, row_col2 = st.columns([1, 6])

            with row_col1:
                st.markdown(f"**Digit {i}**")

            with row_col2:
                st.progress(float(s))