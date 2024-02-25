
import streamlit as st
import openai
import whisper
import os

# Read API key from a file
with open('API_KEY.txt', 'r') as file:
    api_key = file.read().strip()
    
    
# Set your OpenAI API key
openai.api_key = api_key


def transcribe_audio(audio_file):
    # Save the uploaded audio file locally
    with open("uploaded_audio.mp3", "wb") as f:
        f.write(audio_file.getbuffer())
    # Transcribe audio to text using Whisper
    model = whisper.load_model("base")
    text = model.transcribe("uploaded_audio.mp3")
    # Remove the temporary audio file
    os.remove("uploaded_audio.mp3")
    return text


def summarize_text(text):
    # Use OpenAI API to generate a summary using gpt-3.5-turbo model
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI text summarizing assistant"},
            {"role": "user",  "content": f"Summarize the following text: {text}"}
        ]
    )
    return completion.choices[0].message.content



def main():
    st.title("Media Summarizer: Extracting Insights from Audio Files")
    st.caption("This app allows you to upload a audio file which you like to summarize.")
    st.write("")
    st.write("")

    # File uploader for audio
    uploaded_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_audio is not None:
        # Transcribe audio to text
        transcribed_text = transcribe_audio(uploaded_audio)
        st.write("Transcribed text:")
        st.text_area("", transcribed_text["text"], height=200)

        # Summarize the transcribed text
        summary = summarize_text(transcribed_text["text"])
        st.write("Summary:")
        st.text_area("", summary, height=200)

if __name__ == "__main__":
    main()
