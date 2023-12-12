# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Deteccion de Plagas en la agricultura Mexicana",
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
# st.title("Deteccion de Plagas en la agricultura Mexicana")
# st.caption('Updload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
# st.caption('Then click the :blue[Detect Objects] button and check the result.')

# Sidebar
st.sidebar.header("Configuración del modelo de aprendizaje automático")

model_types_available = ['Deteccion', 'OtraTarea', 'OtraTarea2']  # Agrega más tareas según sea necesario
model_type = st.sidebar.multiselect("Seleccionar tarea", model_types_available, default=['Deteccion'])




# Model Options
#model_type = st.sidebar.radio(
#    "Seleccionar tarea", ['Deteccion' ])


confidence = float(st.sidebar.slider(
    "Seleccione la confianza del modelo", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Deteccion':
    model_path = Path(settings.DETECTION_MODEL)
#elif model_type == 'Segmentation':
#    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Imagen/Config")
source_radio = st.sidebar.radio(
    "Seleccione Fuente", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Elige una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagen Cargada",
                         use_column_width=True)
        except Exception as ex:
            st.error("Se produjo un error al abrir la imagen.")
            st.error(ex)

    with col2:        
            if st.sidebar.button('Detectar plaga'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Resultados de la detección"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

#elif source_radio == settings.VIDEO:
#    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

#elif source_radio == settings.YOUTUBE:
#   helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

