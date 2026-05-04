import streamlit as st
from transformers import pipeline
from PIL import Image

# function part
@st.cache_resource
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text
@st.cache_resource
def text2story(text):
    story_generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = f"Write a short story for a 3 to 10-year-old kid about: {text}. The story should be sweet and simple."
    story_text = story_generator(prompt, max_length=100, min_length=50)[0]['generated_text']
    return story_text
@st.cache_resource
def text2audio(story_text):
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_generator(story_text)
    return audio_data

# main part

st.subheader("Kids Story Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file and st.button("Generate Story"):
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    
    st.write("1. Extracting text from image...")
    text = img2text(image)
    st.write(f"Caption: {text}")
    
    st.write("2. Generating story...")
    story = text2story(text)
    st.write(f"Story: {story}")
    
    st.write("3. Generating audio...")
    audio = text2audio(story)
    st.audio(audio["audio"], sample_rate=audio["sampling_rate"])
