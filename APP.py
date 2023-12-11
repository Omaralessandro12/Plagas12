import streamlit as st
import PIL

st.title("Detección de Plagas en la agricultura Mexicana")
st.write ("APLICACION PARA LA DETECCIÓN DE INSECTOS E ACAROS EN LA AGRICULTURA MEXICANA ")

# x = st.slider ("Selecciona un valor ")
# st.write (x, " al cuadrado es ", x*x)


# Barra lateral
st.sidebar.header("Configuración del modelo de aprendizaje automático")


genres = st.sidebar.multiselect(
    "Seleccione las modelos Detección",
    ["YoloV8",  "ResNet50"],
    default=None,
)


st.sidebar.header("Buscar Imagen")


Carimagen = None
Carimagen = st.sidebar.file_uploader(
        "Elige una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

if Carimagen:
   uploaded_image = PIL.Image.open(Carimagen)
   st.image(Carimagen, caption="Imagen Cargada",
        use_column_width=True)




st.sidebar.button('DETECTAR PLAGA')

