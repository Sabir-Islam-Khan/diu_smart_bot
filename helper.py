from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, LLMChain
import nest_asyncio
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
import sitemap
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain 

nest_asyncio.apply()




loader = UnstructuredURLLoader(urls=sitemap.Urls)
data = loader.load()


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

docs = text_splitter.split_documents(data)


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key="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", model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", #similarity
    search_kwargs={'k': 4}
)

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.01, "max_length": 128,"max_new_tokens":2048}
)



#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)






chain = RetrievalQAWithSourcesChain(retriever=retriever, combine_documents_chain=load_qa_chain(llm= llm))


def getInformation(query):
    prompt = f"""
        <|system|>
        You are an AI assistant that follows instruction extremely well.
        Please be truthful and give direct answers
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
    """
    data = chain({"question" : "You are a chatBot of Daffodil Interntional University. You have all the data you need. Answer the questions with context like you are talking to a human. Question is : " + query})
    return data
    

data = getInformation("What is the name of the Dean of FSIT?")

print(data["answer"])
