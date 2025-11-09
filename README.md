## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Researchers often need quick and accurate insights from multiple research papers. Manually searching and summarizing such documents is time-consuming and inefficient. This project aims to build an intelligent retrieval agent using LlamaIndex that can process multiple documents, extract key information, and generate concise, fact-based summaries to assist users in finding relevant knowledge efficiently.

### DESIGN STEPS:

#### STEP 1:
Collect and preprocess multiple research documents (PDFs, text, etc.).

#### STEP 2:
Extract clean text and metadata from each document.

#### STEP 3:
Generate embeddings for document chunks using LlamaIndex.

#### STEP 4:
Build and store a searchable vector index.

#### STEP 5:
Retrieve relevant chunks for user queries.

#### STEP 6:
Generate concise, accurate responses with citations using LLM.

### PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/attachment?id=Yh0a6Xpey6&name=pdf",
    "https://openreview.net/attachment?id=MWCuvhSFPI&name=pdf",
    "https://openreview.net/attachment?id=vsaEOFOUyY&name=pdf",
]

papers = [
    "RISeg.pdf",
    "Online_3D_Edge_Reconstructi.pdf",
    "KnotDLO_Toward_Interpretabl.pdf"
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "What is the core problem addressed by this paper, "
    "and what existing limitations in the current state-of-the-art motivated this work?"
)
response = agent.query("How is fairness, robustness, or interpretability addressed within the context of the paper's main contribution?")
print(str(response))
urls = [
    "https://openreview.net/attachment?id=Yh0a6Xpey6&name=pdf",
    "https://openreview.net/attachment?id=MWCuvhSFPI&name=pdf",
    "https://openreview.net/attachment?id=vsaEOFOUyY&name=pdf",
    "https://openreview.net/attachment?id=s86mu1ovz4&name=pdf",
    "https://openreview.net/attachment?id=1mwJlHsS19&name=pdf"
]

papers = [
    "RISeg.pdf",
    "Online_3D_Edge_Reconstructi.pdf",
    "KnotDLO_Toward_Interpretabl.pdf",
    "Incorporating_Foundation_Model.pdf",
    "Distilling_Semantic_Feature.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
tools = obj_retriever.retrieve(
    "Why is the learning rate a critical hyperparameter in gradient descent optimization?"
)
tools[2].metadata
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "How does the choice of loss function impact the training objective of a model?"
    "Why is cross-validation superior to a single train-test split for generalization estimation?"
)
print(str(response))
response = agent.query(
    "Compare and contrast the IEEE papers"
    "Analyze the approach in each paper first. "
)
```

### OUTPUT:

<img width="1034" height="810" alt="image" src="https://github.com/user-attachments/assets/48416500-e693-44e4-b017-1c85c0afcabc" />

### RESULT:
The multidocument retrieval agent was successfully designed and implemented using LlamaIndex. It efficiently extracted and synthesized information from multiple research papers, providing concise, relevant, and accurate responses to diverse user queries.
