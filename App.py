import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import tempfile

# Cargar modelos de Whisper y TTS
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
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

# Configuración de grabación
DURATION = 5  # Duración de la grabación en segundos
SAMPLE_RATE = 16000  # Frecuencia de muestreo en Hz

# Captura de audio con Sounddevice
def record_audio(duration, sample_rate):
    st.info("Grabando... Habla ahora.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()  # Esperar a que termine la grabación
    return audio

# Interfaz de usuario
st.title("Traductor en Tiempo Real (Inglés/Chino a Español)")

src_lang = st.selectbox("Idioma de origen", ["en (Inglés)", "zh (Chino)"])
src_lang_code = "en" if "Inglés" in src_lang else "zh"
tgt_lang_code = "es"

if st.button("Iniciar Grabación y Traducción"):
    # Grabación de audio
    audio_data = record_audio(DURATION, SAMPLE_RATE)
    
    # Guardar el audio temporalmente
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data.tobytes())
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
