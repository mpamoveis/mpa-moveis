"""
build_index.py — MPA Móveis
Processa os PDFs das NRs e gera o índice FAISS.
Roda UMA VEZ na máquina local. O resultado é commitado no GitHub.

Uso:
    python scripts/build_index.py
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_DIR      = Path("data/pdfs")
INDEX_DIR    = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150


def extrair_texto(caminho: Path) -> str:
    reader = PdfReader(str(caminho))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def detectar_nr(caminho: Path) -> str:
    nome = caminho.stem.upper()
    for nr in ["NR-01","NR-17","NR-18","NR-21","NR-24","NR-31"]:
        if nr.replace("-","") in nome.replace("-",""):
            return nr
    return caminho.stem


def dividir_chunks(texto: str, nr: str) -> list[dict]:
    padrao = re.compile(r"\n(\d+\.\d[\d.]*\s)")
    posicoes = [m.start() for m in padrao.finditer(texto)]

    blocos = []
    if posicoes:
        for i, ini in enumerate(posicoes):
            fim = posicoes[i+1] if i+1 < len(posicoes) else len(texto)
            b = texto[ini:fim].strip()
            if b:
                blocos.append(b)
    else:
        ini = 0
        while ini < len(texto):
            blocos.append(texto[ini:ini+CHUNK_SIZE])
            ini += CHUNK_SIZE - CHUNK_OVERLAP

    chunks, buf = [], ""
    for bloco in blocos:
        if len(buf) + len(bloco) < CHUNK_SIZE:
            buf += "\n" + bloco
        else:
            if buf.strip():
                chunks.append({"nr": nr, "texto": buf.strip()})
            buf = buf[-CHUNK_OVERLAP:] + "\n" + bloco
    if buf.strip():
        chunks.append({"nr": nr, "texto": buf.strip()})
    return chunks


def main():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF em {PDF_DIR}.")
        return

    print(f"{len(pdfs)} PDFs encontrados.\n")
    todos = []
    for pdf in pdfs:
        print(f"Processando {pdf.name}...")
        nr     = detectar_nr(pdf)
        texto  = extrair_texto(pdf)
        chunks = dividir_chunks(texto, nr)
        todos.extend(chunks)
        print(f"  → {len(chunks)} chunks")

    print(f"\nTotal: {len(todos)} chunks. Gerando embeddings...")
    model      = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        [c["texto"] for c in todos],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "nrs.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Índice gerado: {index.ntotal} vetores, dim={dim}")


if __name__ == "__main__":
    main()
