from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

def load_pdf(path:Path)-> list[Document]:
    """
    One document per page, so every chunk remembers its page number.
    """
    reader=PdfReader(str(path))
    return [
        Document(page_content=page.extract_text() or "", metadata={"page": index +1})

        for index, page in enumerate(reader.pages)
    ]

def load_text(path:Path)-> list[Document]:
    text=path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={})]

def load_file(path: Path)-> list[Document]:
    suffix=path.suffix.lower()

    if suffix==".pdf":
        docs=load_pdf(path)
    elif suffix in [".txt", ".md"]:
        docs=load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Drop blank pages(e.g scanned PDFs with no text layer)
    return [d for d in docs if d.page_content.strip()]
