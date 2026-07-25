"""In-memory loaders for the supported office document formats."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
from docx import Document
from pypdf import PdfReader

from pegasus_rag.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ScannedDocumentError,
    UnsupportedFormatError,
)
from pegasus_rag.models import RawSection

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".xlsx"}


def _document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _clean(value: object) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def validate_upload(filename: str, data: bytes, max_size_mb: int) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        accepted = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(f"Formato no permitido. Usa uno de: {accepted}.")
    if not data:
        raise EmptyDocumentError(f"{filename} está vacío.")
    if len(data) > max_size_mb * 1024 * 1024:
        raise CorruptDocumentError(f"{filename} supera el límite de {max_size_mb} MB.")
    return extension


def load_document(
    filename: str,
    data: bytes,
    *,
    max_size_mb: int = 10,
    source_url: str | None = None,
) -> list[RawSection]:
    extension = validate_upload(filename, data, max_size_mb)
    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".csv": _load_csv,
        ".xlsx": _load_xlsx,
    }
    try:
        sections = loaders[extension](filename, data, source_url)
    except (EncryptedDocumentError, EmptyDocumentError, ScannedDocumentError):
        raise
    except Exception as exc:
        raise CorruptDocumentError(
            f"No se pudo leer {filename}: archivo corrupto o inválido."
        ) from exc
    if not sections:
        raise EmptyDocumentError(f"{filename} no contiene texto o datos utilizables.")
    return sections


def load_path(
    path: Path, *, max_size_mb: int = 50, source_url: str | None = None
) -> list[RawSection]:
    return load_document(
        path.name,
        path.read_bytes(),
        max_size_mb=max_size_mb,
        source_url=source_url,
    )


def _load_pdf(filename: str, data: bytes, source_url: str | None) -> list[RawSection]:
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise EncryptedDocumentError(f"{filename} está protegido con contraseña.") from exc
        if not unlocked:
            raise EncryptedDocumentError(f"{filename} está protegido con contraseña.")

    document_id = _document_id(data)
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            sections.append(
                RawSection(text, filename, f"Página {page_number}", document_id, source_url)
            )
    if not sections:
        raise ScannedDocumentError(
            f"{filename} no contiene texto extraíble. El OCR no está incluido en esta versión."
        )
    return sections


def _load_docx(filename: str, data: bytes, source_url: str | None) -> list[RawSection]:
    document = Document(BytesIO(data))
    document_id = _document_id(data)
    sections: list[RawSection] = []
    heading = "Sin sección"
    for number, paragraph in enumerate(document.paragraphs, start=1):
        text = _clean(paragraph.text)
        if not text:
            continue
        if paragraph.style and paragraph.style.name.lower().startswith("heading"):
            heading = text
        sections.append(
            RawSection(
                text,
                filename,
                f"Sección {heading}, párrafo {number}",
                document_id,
                source_url,
            )
        )
    return sections


def _rows_to_sections(
    frame: pd.DataFrame,
    filename: str,
    document_id: str,
    source_url: str | None,
    *,
    sheet: str | None = None,
    block_size: int = 20,
) -> list[RawSection]:
    if frame.empty and len(frame.columns) == 0:
        return []
    frame = frame.fillna("")
    rows = []
    for row_index, row in frame.iterrows():
        values = [f"{column}: {_clean(value)}" for column, value in row.items() if _clean(value)]
        if values:
            rows.append((int(row_index) + 2, "; ".join(values)))
    sections = []
    for start in range(0, len(rows), block_size):
        block = rows[start : start + block_size]
        start_row, end_row = block[0][0], block[-1][0]
        prefix = f"Hoja {sheet}, " if sheet else ""
        location = f"{prefix}filas {start_row}-{end_row}"
        text = "\n".join(f"Fila {number}: {value}" for number, value in block)
        sections.append(RawSection(text, filename, location, document_id, source_url))
    return sections


def _load_csv(filename: str, data: bytes, source_url: str | None) -> list[RawSection]:
    document_id = _document_id(data)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(data), encoding=encoding)
            return _rows_to_sections(frame, filename, document_id, source_url)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def _load_xlsx(filename: str, data: bytes, source_url: str | None) -> list[RawSection]:
    document_id = _document_id(data)
    workbook = pd.ExcelFile(BytesIO(data), engine="openpyxl")
    sections = []
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        sections.extend(
            _rows_to_sections(frame, filename, document_id, source_url, sheet=str(sheet))
        )
    return sections
