from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import streamlit as st
import asyncio
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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

# Write Agent
Translator = Agent(
    name='Translator Agent',
    instructions="""You are a Translator agent. Translate the Paragraph that user will give to you to the language user says to you."""
)

st.set_page_config(page_title="AI Translator", layout="centered")
st.title("🌍 AI Translator Agent")

text_input = st.text_area("✏️ Enter text to translate:")
target_lang = st.text_input("🌐 Target language (e.g., English, French, Urdu):")


async def translate_text():
    return await Runner.run(
        Translator,
        input=f"{text_input}. Translate it into {target_lang}",
        run_config=config,
    )

if st.button("Translate"):
    if text_input and target_lang:
        with st.spinner("Translating..."):
            response = asyncio.run(translate_text())
            st.success("Translation complete!")
            st.write("**🌐 Translation:**")
            st.markdown(response.final_output)
    else:
        st.warning("Please enter both text and target language.")


st.info("💡 You can also give instructions in the language box about the translation.")


