from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from hub import compiled_agent
from hub import State


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask_agent")
async def ask_agent(query: str):
    """
    This endpoint is used to ask the agent a question.
    The agent will process the question and return the answer.
    """
    state = State(
        query=query,
        refined_query="",
        citations=[],
        corrected_query="",
        summary="",
        attempted_retrieval=0,
        evaluation_result="",
    )
    
    # Run the compiled agent with the state object
    result = compiled_agent.invoke(state)
    
    # Convert the result back to a dictionary
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)