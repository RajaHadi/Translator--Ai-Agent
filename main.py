import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load Gemini API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Validate API key
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found. Set it in your .env file.")
    st.stop()

# Set up external client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Use Gemini 2.0 Flash model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define your AI agent (Translator, Writer, etc.)
translator_agent = Agent(
    name="Translator",
    instructions="""
You are a multilingual assistant. You can translate text to any language, explain meanings, 
or simplify complex sentences based on the user's instruction.
"""
)

# --- Streamlit UI ---
st.set_page_config(page_title="🌍 AI Translator", layout="centered")
st.title("🌐 Gemini Translator Agent")

text_input = st.text_area("✏️ Enter word / sentence / paragraph:")
target_lang = st.text_input("📋 What to do? (e.g., Translate to Urdu, Explain in French, etc.)")

st.markdown("""
> 💡 Example Instructions:
> - "Translate into Urdu"
> - "Explain in simple English"
> - "Tell me its meaning in French"
""")

# Handle Button Click
if st.button("🚀 Translate"):
    if text_input.strip() and target_lang.strip():
        with st.spinner("🔄 Translating..."):
            try:
                # Use sync call instead of async
                full_input = f"{text_input.strip()}. {target_lang.strip()}"
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
