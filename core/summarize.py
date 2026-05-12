from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", 
                         mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.2)


def split_transcript(transcript: str) -> list:
    """Split a long transcript into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=200)
    
    return text_splitter.split_text(transcript)

def summarize_transcript(transcript: str) -> str:
    """Summarize a transcript using the Mistral AI model."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that summarizes meeting transcripts concisely."),
        ("human", "Summarize the following transcript:\n\n{text}")
    ])
    
    map_chain = prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    # Combine chunk summaries into a final summary
    combined_summary = "\n\n".join(chunk_summaries)
    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that combines summaries into a concise final summary."),
        ("human", "{text}")
    ])
    combine_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combine_prompt |
        llm| StrOutputParser()
    )

    return combine_chain.invoke(combined_summary)

def generate_title(transcript: str) -> str:
    """Generate a concise title for the meeting based on the transcript."""
    llm = get_llm()
    title_chain =(
        RunnablePassthrough()| RunnableLambda(lambda x: {"text": x}) | 
        ChatPromptTemplate.from_messages([
        ("system", "based on the meeting transcript,generate a concise professional meeting title."),
        ("human", "{text}"),
    ])
    
    | llm | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])  # Use the first 2000 characters for title generation
