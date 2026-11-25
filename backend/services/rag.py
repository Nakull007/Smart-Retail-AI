from __future__ import annotations

import io
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import DocumentChunk
from backend.services.embeddings import embed_texts


def extract_document_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with fitz.open(stream=content, filetype="pdf") as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if suffix == ".docx":
        document = DocxDocument(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    raise ValueError("Only PDF, DOCX, TXT and MD files are supported.")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(clean):
            break
        start = end - overlap
    return chunks


def ingest_document(db: Session, filename: str, content: bytes) -> int:
    text = extract_document_text(filename, content)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No readable text was found in the uploaded document.")
    vectors = embed_texts(chunks)
    safe_source = Path(filename).name[:250]
    # Replace previously ingested chunks from the same filename, so re-upload is idempotent.
    db.query(DocumentChunk).filter(DocumentChunk.source == safe_source).delete(synchronize_session=False)
    db.add_all(
        [
            DocumentChunk(source=safe_source, chunk_index=index, content=chunk, embedding=vector)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
    )
    db.commit()
    return len(chunks)


def retrieve_chunks(db: Session, query: str, top_k: int = 5) -> list[dict]:
    vector = embed_texts([query])[0]
    rows = db.scalars(
        select(DocumentChunk)
        .order_by(DocumentChunk.embedding.cosine_distance(vector))
        .limit(top_k)
    ).all()
    return [
        {
            "source": row.source,
            "chunk_index": row.chunk_index,
            "content": row.content,
        }
        for row in rows
    ]
