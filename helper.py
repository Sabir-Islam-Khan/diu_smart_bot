from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import sitemap as sitemap
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain 
from langchain import OpenAI 
import streamlit as st
load_dotenv()

#loaders = UnstructuredURLLoader(urls=sitemap.Urls)
#data = loaders.load()

#text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

#docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings() 

#vectorstore = FAISS.from_documents(docs, embeddings)

#vectorstore.save_local("vectorstore")

x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

retriever = x.as_retriever()
llm = OpenAI(temperature=0.0)

chain = RetrievalQAWithSourcesChain(retriever=retriever, combine_documents_chain=load_qa_chain(llm= llm))


def get_information(question):
    data = chain({"question" : "You are a chatBot of Daffodil Interntional University. You have all the data you need. Answer the questions with context like you are talking to a human. Question is : " + question})
    return data


def main():
    st.title("Daffodil Smart Bot")

    text_input = st.text_input("What do you want to know?")
    
    if st.button("Get Information"):
        if text_input.strip():
            data = get_information(text_input)
            if "answer" in data:
                answer = data["answer"]
                
                st.success( answer)
                
            else:
                st.error("No answer found for the query.")
        else:
            st.warning("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
