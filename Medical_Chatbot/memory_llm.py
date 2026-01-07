import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# -------------------------
# Environment Setup
# -------------------------
load_dotenv()  # Loads HF_TOKEN from .env file

# -------------------------
# Configuration
# -------------------------
# Current model: flan-t5-base (Good for Q&A, lightweight)
# If you want a smarter chatbot later, change this to: "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MODEL_ID = "google/flan-t5-base"

# IMPORTANT: T5 is a text2text model. 
# If you swap to a model like Llama or Mistral later, change this to "text-generation".
MODEL_TASK = "text2text-generation"

# -------------------------
# Load LLM
# -------------------------
def load_model(repo_id):
    """
    Initializes the HuggingFaceEndpoint with the correct task.
    """
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task=MODEL_TASK,  # <--- This fixes the StopIteration error
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HF_TOKEN") # Explicitly passing just to be safe
    )
    return llm

# -------------------------
# Prompt Template
# -------------------------
CUSTOM_PROMPT = """
Use the pieces of information provided in the context to answer the user's question.
If you do not know the answer, say that you do not know.
Do not fabricate information.
Only answer using the given context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_qa_chain():
    """
    Sets up the RetrievalQA chain.
    """
    # 1. Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Vector Store
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"The path '{DB_FAISS_PATH}' does not exist. Run ingest script first.")

    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # 3. LLM
    llm = load_model(DEFAULT_MODEL_ID)

    # 4. QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt(CUSTOM_PROMPT)
        }
    )
    return qa_chain

# -------------------------
# Run Loop (Optional for CLI usage)
# -------------------------
if __name__ == "__main__":
    qa_chain = get_qa_chain()
    print("Medical Chatbot is ready! (Type 'exit' to quit)")
    
    while True:
        user_query = input("\nUser Query: ")
        
        if user_query.lower() in ["exit", "quit"]:
            break
            
        if not user_query.strip():
            continue

        try:
            # RetrievalQA expects the input key to be "query"
            response = qa_chain.invoke({"query": user_query})

            print("\n------------------------------------------------")
            print("Answer:", response["result"])
            print("------------------------------------------------")
            
            # Optional: Print sources if you want to verify where data came from
            # for doc in response["source_documents"]:
            #     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")