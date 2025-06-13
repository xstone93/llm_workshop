import llm_workshop.streamlit as st
from openai import OpenAI
from typing import List
import requests
import json

# Set OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "YOUR_API_KEY_HERE")

# Generic function to call OpenRouter-compatible models (Mistral, DeepSeek, etc.)
def call_openrouter_model(model_name, messages, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "IT:U Chatbot",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "temperature": 0.7
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Sidebar – chatbot configuration
st.sidebar.title("🔧 Chatbot Configuration")

with st.sidebar.form("config_form"):
    system_prompt = st.text_area("System Prompt", "You are a helpful multimodal assistant. If the user asks for something visual, respond with: 'Please generate an image of: <prompt>'.")
    custom_prompt = st.text_area("Custom Prompt (Optional)", "")
    model = st.selectbox("Model", [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o",
        "deepseek/deepseek-r1-0528-qwen3-8b:free"
    ])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 50, 1000, 500, 50)
    play_audio = st.checkbox("🔊 Enable Text-to-Speech for assistant replies", value=False)
    apply_clicked = st.form_submit_button("✅ Apply Configuration")

# First-time setup (silent, no message)
if "initialized" not in st.session_state:
    st.session_state.messages = []
    st.session_state.initialized = True

# Reset and reinitialize when Apply button is clicked
if apply_clicked:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_prompt})
    if custom_prompt:
        st.session_state.messages.append({"role": "user", "content": custom_prompt})
    st.session_state.play_audio = play_audio
    st.success("Configuration applied. Chat reset.")

# Store current play_audio setting
if "play_audio" not in st.session_state:
    st.session_state.play_audio = True

# Main UI
st.title("💬 IT:U Chatbot")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("image_url"):
                st.image(msg["image_url"], caption="Generated Image")
            else:
                st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message here")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            if model.startswith("mistral") or model.startswith("deepseek"):
                openrouter_key = st.secrets.get("MISTRAL_API_KEY", "your-openrouter-key")
                reply = call_openrouter_model(model, st.session_state.messages, openrouter_key)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                reply = response.choices[0].message.content

            # Step 2: Check for image trigger pattern
            if reply.lower().startswith("please generate an image of:"):
                prompt = reply.split(":", 1)[1].strip()
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = image_response.data[0].url
                st.chat_message("assistant").image(image_url, caption=f"Generated: {prompt}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "image_url": image_url
                })
            else:
                st.chat_message("assistant").markdown(reply)

                # Optional text-to-speech playback
                if st.session_state.play_audio:
                    try:
                        audio_response = client.audio.speech.create(
                            model="tts-1",
                            voice="nova",
                            input=reply
                        )
                        audio_bytes = audio_response.read()
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Could not generate audio: {e}")

                st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            error_msg = f"Error: {e}"
            st.chat_message("assistant").markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer (dark-mode friendly and centered)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 0.75em;
        font-size: 0.85em;
        text-align: center;
        z-index: 100;
        color: #bbb;
    }
    .block-container {
        padding-bottom: 4em;
    } 
    </style>
    <div class="footer">
        🛠️ Built with ❤️ using Streamlit by Alexander Steinmaurer · 2025
    </div>
    """,
    unsafe_allow_html=True
)
