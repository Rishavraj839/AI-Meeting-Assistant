from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever
import os


def get_llm():
    return ChatMistralAI(model="mistral-small-latest", 
                         mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.2)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert meeting assistant"
         that answers questions based on meeting transcripts only.
        If the answer is not found in the transcript, respond with Answer not found in transcript.
        context from meeting transcript: {context}"""),
        ("human", "{question}"),
    ])
    #full lcel rag pipeline
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
          "question": RunnablePassthrough()} |
        prompt | llm | StrOutputParser()    
    )
    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert meeting assistant"
         that answers questions based on meeting transcripts only.
        If the answer is not found in the transcript, respond with Answer not found in transcript.
        context from meeting transcript: {context}"""),
        ("human", "{question}"),
    ])
    #full lcel rag pipeline
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
          "question": RunnablePassthrough()} |
        prompt | llm | StrOutputParser()    
    )
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    """Ask a question to the RAG chain and get an answer."""
    print(f"Asking question: {question}")
    return rag_chain.invoke(question)