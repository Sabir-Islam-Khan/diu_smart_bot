from dotenv import load_dotenv
#from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
import sitemap as sitemap
from langchain.text_splitter import CharacterTextSplitter
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain 
from langchain import OpenAI 
import glob
import textract
import os
from langchain_community.document_loaders import TextLoader
import time

load_dotenv()


try:
   loader = UnstructuredMarkdownLoader("all_data.md")
except Exception as e:
   print("error in loaded.")
   print(e)

data = loader.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=50)

try:
   # Attempt to split documents
   docs = text_splitter.split_documents(data)
except Exception as e:
  #  Handle any exceptions
   print(f"Error occurred while splitting documents: {e}")
   docs = []



embeddings = OpenAIEmbeddings() 
# Define the texts you want to add to the FAISS instance
texts = ["Sabir Islam is magician", "Sohan is a genius"]

# Use the `from_texts` class method of the FAISS class to initialize an instance and add the texts and their corresponding embeddings
vectorstore = FAISS.from_texts(texts, embeddings)

i = 1
t = 1
v = 1
for d in docs:
    vectorstore.add_documents([d])
    print("embedding chunk ", i)
    i += 1
    if i % 100 == 0:

        print("Sleeping...", t)
        t += 1
        time.sleep(60)
    if i %  1000 == 0:
      print("saving vectorstore round ", v)
      v += 1
      vectorstore.save_local("vectorstore")

vectorstore.save_local("vectorstore")


# x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# retriever = x.as_retriever()
# llm = OpenAI(temperature=0.0, model_name="gpt-3.5-turbo-0125", )

# print("llm model is")
# print(llm)
# chain = RetrievalQAWithSourcesChain(retriever=retriever, combine_documents_chain=load_qa_chain(llm= llm))


# def get_information(question):
#     data = chain({"question" : "You are a chatbot of daffodil interntional university. Students ask you about their query and you answer accordingly. Here is a question" + question})
#     return data

# dummy = get_information("What is the total cost of undergraduate in computer science and engineering")
# print(dummy)