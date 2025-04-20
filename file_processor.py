from typing import List
from langchain.schema import Document
from env import GEMINI_API_KEY
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from custom_loader import GeminiLoader

OCR_PROMPT = ("Act like a text scanner. Extract text as it is without analyzing it and without summarizing it. "
              "Treat all images as a whole document and analyze them accordingly. Think of it as a document with "
              "multiple pages, each image being a page. Understand page-to-page flow logically and semantically.")
GEMINI_MODEL = 'gemini-2.0-flash-001'


class ObsidianVault:
    def __init__(self, vault_path):
        # Obsidian vault directory
        self._vault_path = vault_path
        # Directory loader for markdown files only
        self._loader = GeminiLoader(self._vault_path, GEMINI_API_KEY, GEMINI_MODEL)
        # Prep Gemini for OCR
        self._gemini = genai.Client(api_key=GEMINI_API_KEY)

        self._vector_store = None
        self._init_model()

        self._pages = []
        self._split_pages = []

    def _init_model(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        self._vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )


    def text_splitter(self, pages) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        self._split_pages = text_splitter.split_documents(pages)
        print(self._split_pages)

    def parse_vault(self):
        self._pages = self._loader.load()

    def add_pages_to_vector_store(self, split_pages):
        self._vector_store.add_documents(documents=split_pages)

    def process_vault(self) -> None:
        self.parse_vault()
        self.text_splitter(self._pages)
        self.add_pages_to_vector_store(self._split_pages)




    # def ocr_pdf(self, file, source) -> Document:
    #     """
    #     OCRs a pdf curates a LangChain Document object
    #     :param file: the pdf to be OCR'ed
    #     :param source: the pdf file path
    #     :return: Document - a LangChain document object
    #     """
    #     # Render page as high-res image
    #     file = self._gemini.files.upload(file=file)
    #     response = self._gemini.models.generate_content(
    #         model=GEMINI_MODEL,
    #         contents=[OCR_PROMPT, file]
    #     )
    #
    #     doc = Document(
    #         page_content=response.text,
    #         metadata={"source": source}
    #     )
    #     return doc  # String containing the OCR result