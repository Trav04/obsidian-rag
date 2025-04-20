import os
import fitz  # PyMuPDF
import pytesseract
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from env import GEMINI_API_KEY
from google import genai
from google.genai import types

from customer_loader import GeminiLoader

OCR_PROMPT = ("Act like a text scanner. Extract text as it is without analyzing it and without summarizing it. "
              "Treat all images as a whole document and analyze them accordingly. Think of it as a document with "
              "multiple pages, each image being a page. Understand page-to-page flow logically and semantically.")
GEMINI_MODEL = 'gemini-2.0-flash-001'


class ObsidianProcessor:
    def __init__(self, vault_path):
        self._vault_path = vault_path

        # Directory loader for markdown files only
        self._loader = DirectoryLoader(vault_path, glob="**/*.md")
        # Prep Gemini for OCR
        self._gemini = genai.Client(api_key=GEMINI_API_KEY)


    def process_vault(self):
        # Process Markdown
        # pages = []
        #
        # # Process markdown pages into langchain document objects
        # self._loader.glob = "**/*.md"  # Set md files only
        # mark_down = self._loader.load()
        # pages.extend(mark_down)  # Add langchan document lsit to pages lits
        #
        # print(pages)
        loader = GeminiLoader(self._vault_path, GEMINI_API_KEY, GEMINI_MODEL)
        print(loader.load())


    def ocr_pdf(self, file, source) -> Document:
        """
        OCRs a pdf curates a LangChain Document object
        :param file: the pdf to be OCR'ed
        :param source: the pdf file path
        :return: Document - a LangChain document object
        """
        # Render page as high-res image
        file = self._gemini.files.upload(file=file)
        response = self._gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[OCR_PROMPT, file]
        )

        doc = Document(
            page_content=response.text,
            metadata={"source": source}
        )
        return doc  # String containing the OCR result


    def process_pdfs(self):
        pdf_docs = []
        for root, _, files in os.walk(self._vault_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    try:
                        doc = fitz.open(pdf_path)
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            text = self.extract_page_content(page)
                            pdf_docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_num + 1,
                                    "type": "pdf"
                                }
                            ))
                        doc.close()
                    except Exception as e:
                        print(f"Error processing {pdf_path}: {str(e)}")
        return pdf_docs

    def extract_page_content(self, page):
        # Try text extraction first
        text = page.get_text()

        # If text is minimal or missing, use OCR
        if len(text.strip()) < 50:  # Threshold for handwritten content
            text += self.ocr_page(page)

        return text

    def process_images(self):
        img_docs = []
        for root, _, files in os.walk(self._vault_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, file)
                    try:
                        text = pytesseract.image_to_string(img_path)
                        img_docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": img_path,
                                "type": "image"
                            }
                        ))
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        return img_docs