import streamlit as st
from googletrans import Translator
import speech_recognition as sr
import tempfile

# Configuración de la página
st.set_page_config(page_title="Traductor con Entrada de Audio", layout="centered")

# Título de la aplicación
st.title("Traductor de Audio a Español")

# Inicializar herramientas
translator = Translator()
recognizer = sr.Recognizer()

# Widget para grabar audio
st.markdown("### Graba un mensaje de voz")
audio_data = st.audio_input("Graba un mensaje para traducir:")

if audio_data is not None:
    
    # Crear un archivo temporal para almacenar el audio grabado
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data.getbuffer())
        temp_audio_path = temp_audio_file.name  # Ruta del archivo temporal

    try:
        st.info("Procesando el audio...")

        # Cargar el archivo temporal con SpeechRecognition
        with sr.AudioFile(temp_audio_path) as source:
            audio_content = recognizer.record(source)  # Leer todo el contenido del audio
            # Convertir el audio a texto
            detected_text = recognizer.recognize_google(audio_content, language="zh-CN")
            st.success(f"Texto detectado: {detected_text}")

            # Traducir el texto detectado
            translation = translator.translate(detected_text, src="zh-cn", dest="es")
            st.success(f"Traducción: {translation.text}")
    except Exception as e:
        st.error(f"Error al procesar el audio: {e}")
else:
    st.warning("Por favor, graba un mensaje de voz para traducir.")

# Información adicional
st.markdown("---")
st.markdown("Aplicación creada con 💙 utilizando [Streamlit](https://streamlit.io/), [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) y [Googletrans](https://pypi.org/project/googletrans/).")
