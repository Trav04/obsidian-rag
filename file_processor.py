from typing import List
from langchain.schema import Document
from env import GEMINI_API_KEY
from google import genai
from google.genai import types

from custom_loader import GeminiLoader

OCR_PROMPT = ("Act like a text scanner. Extract text as it is without analyzing it and without summarizing it. "
              "Treat all images as a whole document and analyze them accordingly. Think of it as a document with "
              "multiple pages, each image being a page. Understand page-to-page flow logically and semantically.")
GEMINI_MODEL = 'gemini-2.0-flash-001'


class ObsidianProcessor:
    def __init__(self, vault_path):
        self._vault_path = vault_path

        # Directory loader for markdown files only
        self._loader = GeminiLoader(self._vault_path, GEMINI_API_KEY, GEMINI_MODEL)
        # Prep Gemini for OCR
        self._gemini = genai.Client(api_key=GEMINI_API_KEY)


    def process_vault(self) -> List[Document]:
        pages = self._loader.load()
        return pages


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