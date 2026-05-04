import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import io
import soundfile as sf
 
 
# function part
 
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text
 
 
# text2story
def text2story(text):
    story_generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = f"Write a short story for a 3 to 10-year-old kid about: {text}."
    output = story_generator(prompt, max_new_tokens=100, min_new_tokens=40)[0]["generated_text"]
    story_text = output[len(prompt):].strip()
    return story_text
 
 
# text2audio
def text2audio(story_text):
    tts = pipeline("text-to-speech", model="Matthijs/mms-tts-eng")
    result = tts(story_text)
 
    # convert numpy array to WAV bytes so st.audio() can play it
    audio_array = np.array(result["audio"])
    if audio_array.dtype != np.int16:
        audio_array = (audio_array / np.max(np.abs(audio_array)) * 32767).astype(np.int16)
    if audio_array.ndim > 1:
        audio_array = audio_array[0]
 
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, result["sampling_rate"], audio_array)
    buf.seek(0)
    audio_data = buf.read()
    return audio_data
 
 
# main part
st.title("Kids Story Generator")
 
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
 
if uploaded_file and st.button("Generate Story"):
    image = Image.open(uploaded_file)
    st.image(image, width=300)
 
    st.write("1. Extracting caption from image...")
    text = img2text(image)
    st.write(f"Caption: {text}")
 
    st.write("2. Generating story...")
    story = text2story(text)
    st.write(f"Story: {story}")
 
    st.write("3. Generating audio...")
    audio = text2audio(story)
    st.audio(audio, format="audio/wav")
