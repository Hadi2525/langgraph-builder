"""This is an automatically generated file. Do not modify it.

This file was generated using `langgraph-gen` version 0.0.3.
To regenerate this file, run `langgraph-gen` with the source `yaml` file as an argument.

Usage:

1. Add the generated file to your project.
2. Create a new agent using the stub.

Below is a sample implementation of the generated stub:

```python
from typing_extensions import TypedDict

from stub import CustomAgent

class SomeState(TypedDict):
    # define your attributes here
    foo: str

# Define stand-alone functions
def RefineAgent(state: SomeState) -> dict:
    print("In node: RefineAgent")
    return {
        # Add your state update logic here
    }


def Retrieve(state: SomeState) -> dict:
    print("In node: Retrieve")
    return {
        # Add your state update logic here
    }


def EvaluateAgent(state: SomeState) -> dict:
    print("In node: EvaluateAgent")
    return {
        # Add your state update logic here
    }


def CorrectAgent(state: SomeState) -> dict:
    print("In node: CorrectAgent")
    return {
        # Add your state update logic here
    }


def SummaryAgent(state: SomeState) -> dict:
    print("In node: SummaryAgent")
    return {
        # Add your state update logic here
    }


def CheckRAG(state: SomeState) -> str:
    print("In condition: CheckRAG")
    raise NotImplementedError("Implement me.")


agent = CustomAgent(
    state_schema=SomeState,
    impl=[
        ("RefineAgent", RefineAgent),
        ("Retrieve", Retrieve),
        ("EvaluateAgent", EvaluateAgent),
        ("CorrectAgent", CorrectAgent),
        ("SummaryAgent", SummaryAgent),
        ("CheckRAG", CheckRAG),
    ]
)

compiled_agent = agent.compile()

print(compiled_agent.invoke({"foo": "bar"}))
"""

from typing import Callable, Any, Optional, Type

from langgraph.constants import START, END
from langgraph.graph import StateGraph


def CustomAgent(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for CustomAgent."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "RefineAgent",
        "Retrieve",
        "EvaluateAgent",
        "CorrectAgent",
        "SummaryAgent",
        "CheckRAG",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("RefineAgent", nodes_by_name["RefineAgent"])
    builder.add_node("Retrieve", nodes_by_name["Retrieve"])
    builder.add_node("EvaluateAgent", nodes_by_name["EvaluateAgent"])
    builder.add_node("CorrectAgent", nodes_by_name["CorrectAgent"])
    builder.add_node("SummaryAgent", nodes_by_name["SummaryAgent"])

    # Add edges
    builder.add_edge(START, "RefineAgent")
    builder.add_edge("SummaryAgent", END)
    builder.add_edge("RefineAgent", "Retrieve")
    builder.add_edge("Retrieve", "EvaluateAgent")
    builder.add_edge("CorrectAgent", "Retrieve")
    builder.add_conditional_edges(
        "EvaluateAgent",
        nodes_by_name["CheckRAG"],
        [
            "CorrectAgent",
            "SummaryAgent",
        ],
    )
    return builder
