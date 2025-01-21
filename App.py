import streamlit as st
import pyaudio
import numpy as np
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import tempfile

# Cargar modelos de Whisper y TTS
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")  # Usa modelos más pequeños para menor latencia
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

# Configuración de captura de audio
CHUNK = 1024  # Tamaño del fragmento de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Frecuencia de muestreo (16kHz para Whisper)

# Captura de audio en tiempo real
def record_audio(stream, duration=5):
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    return np.hstack(frames)

# Interfaz de usuario
st.title("Traductor en Tiempo Real (Inglés/Chino a Español)")

src_lang = st.selectbox("Idioma de origen", ["en (Inglés)", "zh (Chino)"])
src_lang_code = "en" if "Inglés" in src_lang else "zh"
tgt_lang_code = "es"

if st.button("Iniciar Traducción en Tiempo Real"):
    st.write("Capturando audio en tiempo real... Habla ahora.")
    st.warning("Procesar audio puede tomar unos segundos después de hablar.")
    
    # Configurar PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    try:
        while True:
            # Capturar audio por fragmentos
            audio_data = record_audio(stream, duration=5)  # Captura 5 segundos de audio
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
    
    except KeyboardInterrupt:
        st.write("Traducción en tiempo real detenida.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
