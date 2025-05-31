import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

import nest_asyncio
nest_asyncio.apply()  # <-- This fixes event loop issues in Streamlit

# Load Gemini API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found. Set it in your .env file.")
    st.stop()

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translator_agent = Agent(
    name="Translator",
    instructions="""
You are a multilingual assistant. You can translate text to any language, explain meanings, 
or simplify complex sentences based on the user's instruction.
"""
)

st.set_page_config(page_title="🌍 AI Translator", layout="centered")
st.title("🌐Translator Agent")

text_input = st.text_area("✏️ Enter word / sentence / paragraph:")
target_lang = st.text_input("📋 What to do? (e.g., Translate to urdu, Explain in fernch(any language), etc.)")

st.markdown("""
> 💡 Example Instructions:
> - "Translate into this languge(any language)"
> - "Explain in simple English"
> - "Tell me its meaning in this language(any language)"
""")

if st.button("🚀 Translate"):
    if text_input.strip() and target_lang.strip():
        with st.spinner("🔄 Translating..."):
            try:
                full_input = f"This is prompt {text_input.strip()}. Do this with it = {target_lang.strip()}"
                response = Runner.run_sync(
                    translator_agent,
                    input=full_input,
                    run_config=config
                )
                st.success("✅ Done!")
                st.markdown("### 🌐 Result:")
                st.write(response.final_output)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please fill in both fields.")
