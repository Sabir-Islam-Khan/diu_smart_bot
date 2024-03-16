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
nest_asyncio.apply()




loader = UnstructuredURLLoader(urls=sitemap.Urls)
data = loader.load()

print("data is \n\n")
print(data)

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)

docs = text_splitter.split_documents(data)

print(docs)

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
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)



#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)


query = "Who is dean of FSIT"

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

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

response = qa.run(prompt)

print(response)

#prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": retriever,  "query": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

#response = rag_chain.invoke("Who is the dean of FSIT")

#print(response)
