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

nest_asyncio.apply()


WEBSITE_URL = "https://sabirislam.me/"

loader = WebBaseLoader(WEBSITE_URL)
loader.requests_per_second = 1
docs = loader.aload()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key="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", model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", #similarity
    search_kwargs={'k': 4}
)

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)

query = "Who is Sabir Islam Khan?"

prompt = f"""
 <|system|>
You are an AI assistant that follows instruction extremely well.
Please be truthful and give direct answers.  Answer questions only from the context that you are given
</s>
 <|user|>
 {query}
 </s>
 <|assistant|>
"""

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)



template = """
 <|system|>
You are an AI assistant that follows instruction extremely well.
Please be truthful and give direct answers
</s>
 <|user|>
 {query}
 </s>
 <|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is the context you are provided is about?")

print(response)
