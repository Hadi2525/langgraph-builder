"""This file was generated using `langgraph-gen` version 0.0.3.

This file provides a placeholder implementation for the corresponding stub.

Replace the placeholder implementation with your own logic.
"""

from typing_extensions import TypedDict
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from stub import CustomAgent
from VectorStore import VectorStore
from utils import format_citations

from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

vector_store = VectorStore()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class State(TypedDict):
    # define your attributes here
    query: str
    citations: list[str]
    corrected_query: str
    refined_query: str
    evaluation_result: Literal["CorrectAgent", "SummaryAgent", "Init"]
    attempted_retrieval: int
    summary: str

class EvaluateAgentState(TypedDict):
    """State for the EvaluateAgent."""
    evaluation_result: str

def FirstCheck(state: State) -> str:
    """This is the first conditional check to make sure the user's input is relevant to the underlying vector store."""
    system = """
    You are a helpful assistant. Your task is to check if the user's query is relevant to async programming and asyncio in Python.
    Given the query, determine if it is relevant to the vector store. The underlying vector store is about a book completely focused on async programming and
    asyncio in Python. This book is written by Caleb. Anything relevant you should return "RefineAgent". Use your best judgment
    if you find the query not relevant return "SummaryAgent".
    """
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "User query:\n {query}\n Relevance check result:\n\n")
        ]
    )
    check_llm = llm.with_structured_output(EvaluateAgentState)
    check_chain = check_prompt | check_llm | StrOutputParser()
    result = check_chain.invoke({"query": state["query"]})
    if result['evaluation_result'] == "SummaryAgent":
        return "SummaryAgent"
    return "RefineAgent"

def RefineAgent(state: State) -> dict:
    """This is an agent based state that refines the user's query."""
    query = state.get('query')
    system = """
    You are a helpful assistant. Your task is to refine the user's query to make it more specific and clear.
    Given the query, list a number of relevant topics and keywords that are closest in context and semantics to the query.
    Only return the refined query with the relevant topics and keywords.
    """
    refine_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "Original query:\n {query}\n Refined query:\n\n")
        ]
    )
    refine_chain = refine_prompt | llm | StrOutputParser()

    result = refine_chain.invoke({"query": query})

    state["refined_query"] = result
    return state


def Retrieve(state: State) -> dict:
    """This is an agent based state that retrieves documents from the vector store.
    args:
        state (State): The current state of the agent.
    returns:
        dict: The updated state with the retrieved documents.
    """

    refined_query = state.get('refined_query')
    try:
        results = vector_store.search(query=refined_query, search_type="vector")
        if results:
            state["attempted_retrieval"] += 1
            state["citations"] = format_citations(results)
        
        return state
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise e


def EvaluateAgent(state: State) -> dict:
    """This is an agent based state that evaluates the retrieved documents.
    The evaluation is based on how well the retrieved documents match the refined query.  
    """
    structured_llm = llm.with_structured_output(EvaluateAgentState)

    system = """
    You are a helpful assistant. Your task is to evaluate the retrieved documents based on the refined query.
    Given the refined query and the retrieved documents, determine if the documents are relevant and useful.
    Carefully analyze the retrieved citations along with the underlying relevance score. Use your best judgment 
    to determine if the documents are relevant and useful.
    If the documents are relevant, return "SummaryAgent". If they are not relevant, return "CorrectAgent".
    """

    evaluate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "Refined query:\n {refined_query}\n Retrieved documents:\n {citations}\n Evaluation result:\n\n")
        ]
    )
    evaluate_chain = evaluate_prompt | structured_llm | StrOutputParser()
    result = evaluate_chain.invoke({
        "refined_query": state["refined_query"],
        "citations": state["citations"]
    })
    state["evaluation_result"] = result
    return state

def CheckRAG(state: State) -> str:
    """Checks to make sure the retrieval was successful and the documents are relevant.
    If the retrieval was not successful, return "CorrectAgent". If the retrieval was successful, return "SummaryAgent".
    """
    if state["attempted_retrieval"] < 3 and state["evaluation_result"] == "CorrectAgent":
        return "CorrectAgent"
    return "SummaryAgent"


def CorrectAgent(state: State) -> dict:
    """This is an agent based state that works on re-evaluating the user's query and attempting to work on refining the query
    and then re-attempting the retrieval.
    """
    system = """
    You are a helpful assistant. Your task is to correct the user'squery based on the retrieved documents.
    You are given the refined query, retrieved documents, and the original query.
    If the refined query is not relevant and the retrieved documents are not useful,
    then identify the gaps in the user's query and suggest improvements.
    Re phrase the query and suggest a new query that has a similar semantics to the original query and
    has the tone akin to the retrieved documents.
    """
    correct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "Original query:\n {query}\n Refined query:\n {refined_query}\n Retrieved documents:\n {citations}\n Corrected query:\n\n")
        ]
    )
    correct_chain = correct_prompt | llm | StrOutputParser()
    result = correct_chain.invoke({
        "query": state["query"],
        "refined_query": state["refined_query"],
        "citations": state["citations"]
    })
    state["corrected_query"] = result

    return state


def SummaryAgent(state: State) -> dict:
    """This is an agent based state that summarizes the retrieved documents.
    The summary is based on the refined query and the retrieved documents.
    """
    system = """
    If the retrieved
    documents are empty or None then simply respond with your general knowledge about the query in a very polite and friendly tone. In
    this case encourage the user to ask question regarding the context of the documents.
    You are a helpful assistant. Your task is to summarize the retrieved documents based on the refined query.
    Given the refined query and the retrieved documents, provide a concise summary of the key points and insights.
    Use the retrieved documents to support your summary and provide a clear and informative response. 

    """
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "Refined query:\n {refined_query}\n Retrieved documents:\n {citations}\n Summary:\n\n")
        ]
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    result = summary_chain.invoke({
        "refined_query": state["refined_query"],
        "citations": state["citations"]
    })
    state["summary"] = result
    return state


agent = CustomAgent(
    state_schema=State,
    impl=[
        ("RefineAgent", RefineAgent),
        ("Retrieve", Retrieve),
        ("EvaluateAgent", EvaluateAgent),
        ("CorrectAgent", CorrectAgent),
        ("SummaryAgent", SummaryAgent),
        ("FirstCheck", FirstCheck),
        ("CheckRAG", CheckRAG),
    ],
)

compiled_agent = agent.compile()


