from pathlib import Path
import PIL
import streamlit as st

# Local Modules
import settings
import helper

st.title("Detección de Plagas en la agricultura Mexicana")
st.write("APLICACION PARA LA DETECCIÓN DE INSECTOS E ACAROS EN LA AGRICULTURA MEXICANA ")

# x = st.slider ("Selecciona un valor ")
# st.write (x, " al cuadrado es ", x*x)

# Barra lateral
st.sidebar.header("Configuración del modelo de aprendizaje automático")

genres = st.sidebar.multiselect(
    "Seleccione las modelos Detección",
    ["YoloV8", "ResNet50"],
    default=None,
)

# Selecting Detection Or Segmentation
model_type = genres[0]  # Selecciona el primer elemento de la lista
if model_type == 'YoloV8':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Buscar Imagen")

Carimagen = None
Carimagen = st.sidebar.file_uploader("Elige una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    try:
        if Carimagen:
            uploaded_image = PIL.Image.open(Carimagen)
            st.image(Carimagen, caption="Imagen Cargada", use_column_width=True)
    except Exception as ex:
        st.error(ex)

with col2:
    if st.sidebar.button('DETECTAR PLAGA'):
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        try:
            with st.expander("Resultados de la detección"):
                for box in boxes:
                    st.write(box.data)
        except Exception as ex:
            st.write("¡Aún no se ha subido ninguna imagen!")
