import streamlit as st
from transformers import pipeline
from PIL import Image

# ==========================================
# Function Part
# ==========================================

@st.cache_resource
def load_img2text_model():
    # Load the image captioning model
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def img2text(image):
    """Processes an uploaded image and generates a text caption."""
    image_to_text_model = load_img2text_model()
    # The pipeline can accept a PIL Image object directly
    text = image_to_text_model(image)[0]["generated_text"]
    return text

@st.cache_resource
def load_text2story_model():
    # Load the text generation model
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

def text2story(text):
    """Generates a 50-100 word story tailored for kids based on the caption."""
    story_generator = load_text2story_model()
    
    # Prompt is designed specifically for the target audience (3-10 year old kids)
    prompt = f"Write a short, fun, and magical story for a 5-year-old kid about: {text}. The story should be sweet and simple."
    
    # Generate story with length constraints (approx 50-100 words)
    story_output = story_generator(
        prompt, 
        max_length=150,  # Limits tokens to keep it around 100 words max
        min_length=50,   # Ensures it reaches at least 50 tokens
        do_sample=True,
        temperature=0.7
    )
    
    story_text = story_output[0]['generated_text']
    
    # Optional: Clean up the output to remove the prompt part if the model repeats it
    if story_text.startswith(prompt):
        story_text = story_text[len(prompt):].strip()
        
    return story_text

@st.cache_resource
def load_text2audio_model():
    # Load the text-to-speech model
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

def text2audio(story_text):
    """Converts the generated story text into an audio format."""
    audio_generator = load_text2audio_model()
    audio_data = audio_generator(story_text)
    return audio_data

# ==========================================
# Main Part
# ==========================================

def main():
    # Set up the page config
    st.set_page_config(page_title="Magic Storyteller", page_icon="🧸", layout="centered")
    st.title("🧸 Turn Your Picture into a Magic Story!")
    st.write("Upload an image, and let's create a fun story for kids!")

    # User uploads an image
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image', use_container_width=True)
        
        # Add a button to trigger the generation process
        if st.button("Generate Story & Audio 🪄"):
            
            # Stage 1: Image to Text (Captioning)
            with st.spinner('Looking at the picture... 🧐'):
                scenario = img2text(image)
                st.info(f"**What I see:** {scenario}")
            
            # Stage 2: Text to Story
            with st.spinner('Writing a magical story... ✍️'):
                story = text2story(scenario)
                st.success(f"**Your Story:**\n\n{story}")
            
            # Stage 3: Story to Audio
            with st.spinner('Recording the story... 🎙️'):
                speech_output = text2audio(story)
                
                audio_array = speech_output["audio"]
                sample_rate = speech_output["sampling_rate"]
                
                st.markdown("### Listen to the Story 🎧")
                st.audio(audio_array, sample_rate=sample_rate)

if __name__ == "__main__":
    main()
