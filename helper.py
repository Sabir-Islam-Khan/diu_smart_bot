from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import sitemap as sitemap
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 

load_dotenv()

loaders = UnstructuredURLLoader(urls=sitemap.Urls)
data = loaders.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings() 

vector_store = FAISS.from_documents(docs, embeddings) 

with open("vector_store.pkl", "wb") as f:
    pickle.dump(vector_store, f)