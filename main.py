import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def detectEmotion(text, lang):
    llm = OpenAI(temperature=0.5)
    promptTemplate = PromptTemplate(
        input_variables=["emotion"],
        template="""Tell me a romantic caption for my Facebook post. Caption type is {text} and language is {lang}. Return only the caption. Nothing else should be included"""
    )
    
    emotion_chain = LLMChain(llm=llm, prompt=promptTemplate)
    
    response = emotion_chain({"text": text, "lang": lang})
    
    return response

def main():
    st.title("Romantic Caption Generator")
    
    text_input = st.text_input("Enter the text for the caption:")
    lang_input = st.text_input("Enter the language for the caption (e.g., en for English):")
    
    if st.button("Generate Caption"):
        if text_input.strip() and lang_input.strip():
            caption = detectEmotion(text_input, lang_input)
            st.write("Generated Caption:", caption)
        else:
            st.warning("Please enter text and language.")

if __name__ == "__main__":
    main()
