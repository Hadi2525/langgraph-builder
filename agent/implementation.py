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
    evaluation_result: Literal["CorrectAgent", "SummaryAgent"]
    attempted_retrieval: int

class EvaluateAgentState(TypedDict):
    """State for the EvaluateAgent."""
    evaluation_result: str

# Define stand-alone functions
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
    print("In condition: CheckRAG")
    raise NotImplementedError("Implement me.")


def CorrectAgent(state: State) -> dict:
    print("In node: CorrectAgent")
    return {
        # Add your state update logic here
    }


def SummaryAgent(state: State) -> dict:
    print("In node: SummaryAgent")
    return {
        # Add your state update logic here
    }


agent = CustomAgent(
    state_schema=State,
    impl=[
        ("RefineAgent", RefineAgent),
        ("Retrieve", Retrieve),
        ("EvaluateAgent", EvaluateAgent),
        ("CorrectAgent", CorrectAgent),
        ("SummaryAgent", SummaryAgent),
        ("CheckRAG", CheckRAG),
    ],
)

compiled_agent = agent.compile()


sample_state = State(
    query="Give me some examples of building async functions in python give full tutorial instructions?",
    citations=[],
    corrected_query="",
    refined_query="",
    evaluation_result="",
    attempted_retrieval=0
)
print(compiled_agent.invoke(sample_state))
