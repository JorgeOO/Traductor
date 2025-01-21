import streamlit as st
from googletrans import Translator

# Configuración de la página
st.set_page_config(page_title="Traductor Chino/Inglés a Español", layout="centered")

# Título de la aplicación
st.title("Traductor Chino/Inglés a Español")

# Inicializar el traductor
translator = Translator()

# Selección del idioma de origen
source_language = st.selectbox("Selecciona el idioma de origen", ["Chino", "Inglés"])
source_lang_code = "zh-cn" if source_language == "Chino" else "en"

# Caja de texto para ingresar el texto a traducir
text_to_translate = st.text_area("Ingresa el texto que deseas traducir:")

# Botón para traducir
if st.button("Traducir"):
    if text_to_translate.strip():
        try:
            # Realizar la traducción
            translation = translator.translate(text_to_translate, src=source_lang_code, dest="es")
            st.success(f"Traducción: {translation.text}")
        except Exception as e:
            st.error(f"Error al traducir: {e}")
    else:
        st.warning("Por favor, ingresa un texto para traducir.")

# Información del pie de página
st.markdown("---")
st.markdown("Aplicación de traducción creada con 💙 utilizando [Streamlit](https://streamlit.io/) y [Googletrans](https://pypi.org/project/googletrans/).")
