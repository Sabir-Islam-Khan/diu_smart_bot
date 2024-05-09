from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
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

load_dotenv()

# def load_data_from_files(directory):
#     all_data = ""
    
#     # Iterate over all files in the directory
#     for file_path in glob.glob(os.path.join(directory, "*")):
#         print("Processing file:", file_path)  # Debug print
        
#         if file_path.endswith(".txt"):
#             # Read text from text files and append to all_data string
#             with open(file_path, "r", encoding="utf-8") as file:
#                 all_data += file.read() + "\n"  # Add a newline separator
#         elif file_path.endswith(".pdf"):
#             # Extract text from PDF files and append to all_data string
#             text = textract.process(file_path, method="pdftotext").decode("utf-8")
#             all_data += text + "\n"  # Add a newline separator
#     print("diu data is")
#     print(all_data)   
#     return all_data




#data = load_data_from_files("diu_data")

try:
   loader = TextLoader("diu_data/combined_file.txt")
except Exception as e:
   print("error in loaded.")
   print(e)

data = loader.load()

print("data is ",  data)
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

try:
   # Attempt to split documents
   docs = text_splitter.split_documents(data)
except Exception as e:
  #  Handle any exceptions
   print(f"Error occurred while splitting documents: {e}")
   docs = []

#Proceed with the rest of your code
print("Docs here is")
print(docs)

embeddings = OpenAIEmbeddings() 

vectorstore = FAISS.from_documents(docs, embeddings)

vectorstore.save_local("vectorstore")

x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

retriever = x.as_retriever()
llm = OpenAI(temperature=0.0, model_name="gpt-3.5-turbo-0125", )

print("llm model is")
print(llm)
chain = RetrievalQAWithSourcesChain(retriever=retriever, combine_documents_chain=load_qa_chain(llm= llm))


def get_information(question):
    data = chain({"question" : "You are a chatbot of daffodil interntional university. Students ask you about their query and you answer accordingly. Here is a question" + question})
    return data

dummy = get_information("all the available scholarships for cse")
print(dummy)
