import os
from typing import List, Dict, Any

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEndpoint

from env import HUGGING_FACE_API
from obsidian_vault import ObsidianVault
from state import State


class RAG:
    def __init__(self, vault_path: str, api_key: str = None, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initialize the RAG system with a vault path and optional API key.

        Args:
            vault_path: Path to the Obsidian vault
            api_key: Hugging Face API key
            model_name: Hugging Face model name to use
        """
        self.vault_path = vault_path
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            raise ValueError("Hugging Face API key must be provided or set as HUGGINGFACE_API_KEY environment variable")

        # Initialize the vault
        self.vault = ObsidianVault(vault_path)
        self.vault.process_vault()
        self.vector_store = self.vault.get_vector_store()

        # Initialize the model
        self.llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
            max_new_tokens=512
        )

        # Get the prompt from LangChain Hub
        self.prompt = hub.pull("rlm/rag-prompt")

        # Initialize the graph
        self._build_graph()

    def retrieve(self, state: State) -> Dict[str, List[Document]]:
        """Retrieve relevant documents from the vector store."""
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State) -> Dict[str, str]:
        """Generate an answer based on the retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response}

    def _build_graph(self):
        """Build the state graph for the RAG system."""
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")

        self.graph = graph_builder.compile()

    def query(self, question: str) -> str:
        """
        Process a question through the RAG system.

        Args:
            question: The question to answer

        Returns:
            The generated answer
        """
        response = self.graph.invoke({"question": question})
        return response["answer"]


if __name__ == "__main__":
    # Replace with your actual API key or set it as an environment variable
    api_key = HUGGING_FACE_API # or None to use environment variable

    # Initialize the RAG system
    rag_system = RAG(
        vault_path="C:\\Users\\tngra\\Documents\\Main",
        api_key=api_key,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"  # Use your preferred model
    )

    # Query the system
    answer = rag_system.query("What tea varieties does Very Good use?")
    print(answer)