from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import sitemap as sitemap
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

loaders = UnstructuredURLLoader(urls=sitemap.Urls)
data = loaders.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

docs = text_splitter.split_documents(data)

print(docs)