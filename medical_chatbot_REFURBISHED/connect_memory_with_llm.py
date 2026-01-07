import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME ="llama-3.3-70b-versatile"

llm=ChatGroq(
        model_name=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
        
    )
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 2: Connect LLM with FAISS and Create chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)


# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("SOURCE DOCUMENTS: ", response["context"])
