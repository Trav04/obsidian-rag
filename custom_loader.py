from typing import AsyncIterator, Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Pattern, Union

from google import genai
from google.genai import types

from langchain_community.document_loaders import ObsidianLoader

OCR_PROMPT = ("Act like a text scanner. Extract text as it is without analyzing it and without summarizing it. "
              "Treat all images as a whole document and analyze them accordingly. Think of it as a document with "
              "multiple pages, each image being a page. Understand page-to-page flow logically and semantically.")


def _parse_md(source):
    with open(source, encoding="UTF-8") as f:
        text = f.read()
    return text


class GeminiLoader(BaseLoader):
    """A custom document loader that uses Gemini to OCR PDFs and images """

    def __init__(self, file_path: str, api_key: str, model: str) -> None:
        """
        Initialises the file path and setups the the Gemini model
        :param file_path: the path to the directory
        :param api_key: Gemini api key
        :param model: Gemini model to be used for parsing
        """
        self.file_path = file_path
        self.api_key = api_key
        self.model = model

        self._gemini = genai.Client(api_key=api_key)

    def _parse_object(self, source) -> str:
        """
        Parses a file object by uploading it to Gemini and returning
        extracted text
        :param source: the file path
        :return: the extracted text
        """
        file = self._gemini.files.upload(file=source)
        response = self._gemini.models.generate_content(
            model=self.model,
            contents=[OCR_PROMPT, file]
        )
        return response.text

    def _parse_image(self, source):
        pass

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        directory = Path(self.file_path)
        for path in directory.iterdir():
            if path.is_file():
                text = ""
                extension = path.suffix
                if extension == ".pdf":
                    text = "this is a pdf"
                    # text = self._parse_object(path)
                elif extension in [".png", ".jpg", ".jpeg"]:
                    text = "this is an image"
                    # text = self._parse_object(path)
                elif extension == ".md":
                    text = _parse_md(path)

                yield Document(page_content=text, metadata={"source": path.as_posix()})
