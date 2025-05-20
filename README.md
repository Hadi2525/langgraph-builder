# LangGraph Builder Agent

LangGraph Builder Agent is a modular Python project designed to facilitate the creation, management, and utilization of language graph-based agents. It provides tools for vector storage, agent orchestration, and integration with external resources, making it suitable for building advanced AI-driven applications.

## Table of Contents
- [LangGraph Builder Agent](#langgraph-builder-agent)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [PDF Ingestion](#pdf-ingestion)
  - [Configuration](#configuration)
  - [Vector Store](#vector-store)
  - [Extending the Agent](#extending-the-agent)
  - [License](#license)

## Features
- Modular agent architecture
- Vector store integration (ChromaDB)
- PDF data ingestion
- Utility functions for agent workflows
- Configurable via JSON and YAML

## Project Structure
```
agent/
  __init__.py         # Package initialization
  hub.py              # Agent hub and orchestration logic
  langgraph.json      # Agent configuration (JSON)
  main.py             # Entry point for running the agent
  spec.yml            # Agent specification (YAML)
  stub.py             # Agent stub/boilerplate
  utils.py            # Utility functions
  VectorStore.py      # Vector store abstraction
  vectorstore/        # Vector store data (ChromaDB)
data/
  pdf/                # PDF files for ingestion
vectorstore/          # ChromaDB vector store files
requirements.txt      # Python dependencies
README.md             # Project documentation
LICENSE               # License file
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd langgraph-builder
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Install the inmem for the `langgraph-cli` to make sure you installed all the dependencies:
```bash
pip install -U "langgraph-cli[inmem]"
```

Set the current directory to the `agent` folder:
```bash
cd agent
```
Run the agent inside LangGraph Studio (locally):
```bash
langgraph dev
```
Set the required values for the input json in the studio tool.


### PDF Ingestion
Place your PDF files in the `data/pdf/` directory. The agent will process and index them into the vector store for retrieval and search.

I used the following book for building the vector store:
[Using Asyncio in Python](https://edu.anarcho-copy.org/Programming%20Languages/Python/using-asyncio-python-understanding-asynchronous.pdf).

## Configuration
- `agent/langgraph.json`: Main configuration for setting up the `LangGraph Studio`.
- `agent/spec.yml`: YAML specification for agent structure

## Vector Store
- Uses ChromaDB for efficient vector storage and retrieval
- Vector data is stored in `agent/vectorstore/` and `vectorstore/`
- `VectorStore.py` provides an abstraction layer for vector operations

## Extending the Agent
- Add new agent logic in `agent/hub.py` or as new modules in `agent/`
- Utility functions can be added to `agent/utils.py`
- Update configuration files as needed for new features

## License
See [LICENSE](LICENSE) for details.