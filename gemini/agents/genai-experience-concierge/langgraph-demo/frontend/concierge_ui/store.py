import uuid
from typing import IO
from pathlib import Path
from typing import TypedDict

import httpx
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from concierge_ui import remote_settings, auth
from concurrent.futures import ThreadPoolExecutor


class PDFChunk(TypedDict):
    page: int
    chunk: int
    content: str


def upload_pdf(
    pdf: UploadedFile,
    chunk_size: int,
    chunk_overlap: int,
    namespace: tuple[str, ...],
    config: remote_settings.StoreConfig,
) -> None:
    pdf_chunks = get_pdf_chunks(pdf, chunk_size, chunk_overlap)

    client = httpx.Client(base_url=str(config.base_url), headers=auth.get_auth_headers(config))
    with ThreadPoolExecutor() as executor:
        for res in executor.map(
            lambda chunk: client.put(
                "/store/items",
                json={
                    "namespace": list(namespace),
                    "key": uuid.uuid4().hex,
                    "value": {config.retrieval_text_field: chunk["content"], "chunk": chunk},
                    "index": [config.retrieval_text_field],
                    "ttl": 60,
                },
            ),
            pdf_chunks,
        ):
            res.raise_for_status()


def get_pdf_chunks(pdf: str | IO | Path, chunk_size: int, chunk_overlap: int) -> list[PDFChunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        PDFChunk(page=page_idx, chunk=chunk_idx, content=chunk)
        for page_idx, page in enumerate(PdfReader(pdf).pages)
        for chunk_idx, chunk in enumerate(text_splitter.split_text(page.extract_text()))
    ]
