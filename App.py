import streamlit as st
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import tempfile

# Cargar modelos
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")  # Modelo Whisper ligero
    tts_model = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False, gpu=False)
    return whisper_model, tts_model

whisper_model, tts_model = load_models()

# Función para traducir texto
def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Interfaz de usuario

st.title("Traductor en Tiempo Real (Inglés/Chino a Español)")

src_lang = st.selectbox("Idioma de origen", ["en (Inglés)", "zh (Chino)"])
src_lang_code = "en" if "Inglés" in src_lang else "zh"
tgt_lang_code = "es"

audio_data = st.audio_input("Graba tu voz", type="wav")  # Componente de grabación

if st.button("Procesar Audio"):
    if audio_data is not None:
        # Guardar archivo de audio temporal
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_data.getvalue())
            temp_audio_path = temp_audio_file.name

        # Transcribir audio
        st.write("Procesando transcripción...")
        transcription = whisper_model.transcribe(temp_audio_path, language=src_lang_code)
        transcribed_text = transcription["text"]
        st.write(f"Texto transcrito: {transcribed_text}")

        # Traducir texto
        st.write("Traduciendo...")
        translated_text = translate_text(transcribed_text, src_lang_code, tgt_lang_code)
        st.write(f"Traducción: {translated_text}")

        # Convertir texto traducido a voz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_audio:
            tts_model.tts_to_file(text=translated_text, file_path=temp_output_audio.name)
            st.audio(temp_output_audio.name, format="audio/wav")
    else:
        st.warning("Por favor, graba un audio antes de procesar.")
