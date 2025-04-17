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
    namespace: tuple[str, ...],
    config: remote_settings.StoreConfig,
) -> None:
    pdf_chunks = get_pdf_chunks(pdf)

    client = httpx.Client(base_url=str(config.base_url), headers=auth.get_auth_headers(config))
    with ThreadPoolExecutor() as executor:
        # wrap in list to join all threads
        list(
            executor.map(
                put_item,
                (namespace for _ in range(len(pdf_chunks))),
                (uuid.uuid4().hex for _ in range(len(pdf_chunks))),
                (
                    {config.retrieval_text_field: chunk["content"], "chunk": chunk}
                    for chunk in pdf_chunks
                ),
                (client for _ in range(len(pdf_chunks))),
            )
        )


def put_item(
    namespace: tuple[str, ...],
    key: str,
    value: dict,
    httpx_client: httpx.Client,
) -> None:
    res = httpx_client.put(
        "/store/items",
        json={"namespace": list(namespace), "key": key, "value": value},
    )
    res.raise_for_status()


def get_pdf_chunks(pdf: str | IO | Path) -> list[PDFChunk]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    return [
        PDFChunk(page=page_idx, chunk=chunk_idx, content=chunk)
        for page_idx, page in enumerate(PdfReader(pdf).pages)
        for chunk_idx, chunk in enumerate(text_splitter.split_text(page.extract_text()))
    ]
