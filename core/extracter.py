#actionable item , decisioms ,questions
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os



def get_llm():
    return ChatMistralAI(model="mistral-small-latest", 
                         mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.2)


def build_chat_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | 
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
    )

def extract_actionable_items(transcript: str) -> str:
    """Extract actionable items from a meeting transcript."""
    system_prompt = (
        "You are an expert meeting analyst that extracts actionable items from meeting transcripts. "
        "Extract all actionable items from the following transcript.\n"
        "- Task Description\n"
        "- Responsible Person (if mentioned)\n"
        "- Deadline if mentioned\n"
        "Format the output as a bullet list.if no actionable items are found, respond with 'No actionable items found.'"

    )
    chain = build_chat_chain(system_prompt)
    return chain.invoke(transcript) 

def extract_key_decisions(transcript: str) -> str:
    """Extract decisions from a meeting transcript."""
    system_prompt = (
        "You are an expert meeting analyst that extracts decisions from meeting transcripts. "
        "Extract all decisions made in the following transcript.\n"
        "Format the output as a bullet list. If no decisions are found, respond with 'No decisions found.'"
    )
    chain = build_chat_chain(system_prompt)
    return chain.invoke(transcript)

def extract_key_questions(transcript: str) -> str:
    """Extract questions from a meeting transcript."""
    system_prompt = (
        "You are an expert meeting analyst that extracts questions from meeting transcripts. "
        "Extract all questions raised in the following transcript.\n"
        "Format the output as a bullet list. If no questions are found, respond with 'No questions found.'"
    )
    chain = build_chat_chain(system_prompt)
    return chain.invoke(transcript)