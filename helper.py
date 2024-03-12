from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import sitemap as sitemap

load_dotenv()

loaders = UnstructuredURLLoader(urls=sitemap.Urls)
data = loaders.load()

