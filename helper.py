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
