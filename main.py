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

text_input = st.text_input("✏️ Enter a word or paragraph:")
target_lang = st.text_area("📋 Enter your instruction (e.g., Translate to Urdu, or Explain meaning in French):")

# 💡 Usage Note
st.markdown("""
> 💡 **Note:** Enter the **word or paragraph** in the first box. In the second box, write what you want — for example:  
> - "Translate it into Urdu"  
> - "Tell me its meaning in French"  
> - "Explain the word in simple English"
""")

# Define async wrapper
async def run_translation():
    return await Runner.run_async(
        Translator,
        input=f"{text_input}. {target_lang}",
        run_config=config
    )

# Handle translate button
if st.button("Translate"):
    if text_input and target_lang:
        with st.spinner("Translating..."):
            response = asyncio.run(run_translation())
            st.success("Translation complete!")
            st.write("**🌐 Result:**")
            st.markdown(response.output)
    else:
        st.warning("Please fill in both the word/paragraph and your instruction.")

