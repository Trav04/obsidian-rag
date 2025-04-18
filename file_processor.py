import os
import fitz  # PyMuPDF
import pytesseract
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
import tempfile


class ObsidianProcessor:
    def __init__(self, vault_path):
        self.vault_path = vault_path

    def process_vault(self):
        # Process Markdown
        # md_loader = DirectoryLoader(self.vault_path, glob="**/*.md")
        # md_docs = md_loader.load()

        # Process PDFs with hybrid text/OCR extraction
        pdf_docs = self.process_pdfs()
        print(pdf_docs)

        # Process Images (standalone images)
        # img_docs = self.process_images()

        # return md_docs + pdf_docs + img_docs

    def process_pdfs(self):
        pdf_docs = []
        for root, _, files in os.walk(self.vault_path):
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

    def ocr_page(self, page, dpi=300):
        # Render page as high-res image
        pix = page.get_pixmap(dpi=dpi)
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, f"temp_page_{page.number}.png")

        try:
            pix.save(image_path)
            ocr_text = pytesseract.image_to_string(image_path)
            os.remove(image_path)
            return f"\n[OCR Result]\n{ocr_text}"
        except Exception as e:
            print(f"OCR failed: {str(e)}")
            return ""

    def process_images(self):
        img_docs = []
        for root, _, files in os.walk(self.vault_path):
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