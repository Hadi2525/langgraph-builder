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

from dotenv import load_dotenv
import os
load_dotenv()



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class State(TypedDict):
    # define your attributes here
    query: str
    citations: list[str]
    corrected_query: str
    refined_query: str
    evaluation_result: Literal["CorrectAgent", "SummaryAgent"]
    attempted_retrieval: int

# Define stand-alone functions
def RefineAgent(state: State) -> dict:
    """This is an agent based state that refines the user's query."""
    query = state.get('query')
    system = """
    You are a helpful assistant. Your task is to refine the user's query to make it more specific and clear.
    Given the query, list a number of relevant topics and keywords that the user might be interested in.
    Give as much insight as possible. Rephrase the query in 5 more ways that clarify the user's intent.
    Keep the length of the refined query to less than 70 words. Your response should have the original query
    in the beginning, followed by the refiened query. The refined query should be the same length as the original query.
    """
    refine_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("ai", "Original query:\n {query}\n Refined query:\n")
        ]
    )
    refine_chain = refine_prompt | llm | StrOutputParser()

    result = refine_chain.invoke({"query": query})

    state["refined_query"] = result
    return state


def Retrieve(state: State) -> dict:
    print("In node: Retrieve")
    return {
        # Add your state update logic here
    }


def EvaluateAgent(state: State) -> dict:
    print("In node: EvaluateAgent")
    return {
        # Add your state update logic here
    }


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


def CheckRAG(state: State) -> str:
    print("In condition: CheckRAG")
    raise NotImplementedError("Implement me.")


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
    query="What is the meaning of life?",
    citations=[],
    corrected_query="",
    refined_query="",
    evaluation_result="",
    attempted_retrieval=0
)
print(compiled_agent.invoke(sample_state))
