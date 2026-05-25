"""
build_index.py — MPA Móveis
Reconstrói o índice FAISS a partir dos PDFs das NRs.

Estratégia de chunking:
- Tenta quebrar pelo padrão de subitens (ex: 24.7.3, 18.5.6)
- Usa overlap de 200 caracteres entre chunks para evitar cortes em itens (a, b, c...)
- Chunk mínimo de 100 caracteres e máximo de 1000 caracteres

Uso:
    python build_index.py
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Configuração ─────────────────────────────────────────────────────────────
PDF_DIR    = Path("data/pdfs")
INDEX_DIR  = Path("data/index")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_MAX  = 2000   # caracteres máximos por chunk
CHUNK_MIN  = 100    # caracteres mínimos — descarta chunks muito pequenos
OVERLAP    = 400    # overlap entre chunks para não perder itens cortados

# Mapeamento nome-do-arquivo → rótulo da NR
NR_LABELS = {
    "nr-01": "NR-01",
    "nr-17": "NR-17",
    "nr-18": "NR-18",
    "nr-21": "NR-21",
    "nr-24": "NR-24",
    "nr-31": "NR-31",
}


# ── Extração de texto do PDF ──────────────────────────────────────────────────
def extrair_texto(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    paginas = []
    for page in reader.pages:
        texto = page.extract_text()
        if texto:
            paginas.append(texto)
    return "\n".join(paginas)


# ── Chunking inteligente com overlap ─────────────────────────────────────────
def chunk_texto(texto: str, nr_label: str) -> list[dict]:
    """
    Divide o texto em chunks respeitando subitens da NR.
    Usa overlap para evitar que itens (a, b, c...) fiquem cortados entre chunks.
    """
    chunks = []

    # Tenta dividir nos pontos de início de subitem (ex: 24.7.3, 18.5.6.1)
    # Padrão: número(s).número(s) no início de linha ou após quebra
    padrao_subitem = re.compile(
        r'(?=\n\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})*(?:\.\d{1,2})?\s)',
        re.MULTILINE
    )

    # Divide nos pontos de subitem
    segmentos = padrao_subitem.split(texto)

    # Reconstrói chunks respeitando CHUNK_MAX com overlap
    buffer = ""
    for seg in segmentos:
        seg = seg.strip()
        if not seg:
            continue

        # Se o segmento cabe no buffer, acumula
        if len(buffer) + len(seg) <= CHUNK_MAX:
            buffer += "\n" + seg if buffer else seg
        else:
            # Salva o buffer atual se for suficientemente grande
            if len(buffer) >= CHUNK_MIN:
                chunks.append({"nr": nr_label, "texto": buffer.strip()})

            # Começa novo buffer com overlap do final do buffer anterior
            overlap_texto = buffer[-OVERLAP:] if len(buffer) > OVERLAP else buffer
            buffer = overlap_texto + "\n" + seg if overlap_texto else seg

    # Salva o último buffer
    if buffer.strip() and len(buffer.strip()) >= CHUNK_MIN:
        chunks.append({"nr": nr_label, "texto": buffer.strip()})

    return chunks


# ── Detecta rótulo da NR pelo nome do arquivo ─────────────────────────────────
def detectar_nr(pdf_path: Path) -> str | None:
    nome = pdf_path.stem.lower()
    for chave, label in NR_LABELS.items():
        if chave in nome:
            return label
    return None


# ── Pipeline principal ────────────────────────────────────────────────────────
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em {PDF_DIR}")
        return

    print(f"PDFs encontrados: {len(pdfs)}")
    for p in pdfs:
        print(f"  - {p.name}")

    # ── 1. Extrai e chunka todos os PDFs ─────────────────────────────────────
    todos_chunks: list[dict] = []

    for pdf_path in pdfs:
        nr_label = detectar_nr(pdf_path)
        if not nr_label:
            print(f"  [AVISO] NR não identificada para: {pdf_path.name} — pulando")
            continue

        print(f"\nProcessando {pdf_path.name} → {nr_label}")
        texto  = extrair_texto(pdf_path)
        chunks = chunk_texto(texto, nr_label)
        todos_chunks.extend(chunks)
        print(f"  Chunks gerados: {len(chunks)}")

    print(f"\nTotal de chunks: {len(todos_chunks)}")

    # ── 2. Gera embeddings ────────────────────────────────────────────────────
    print("\nCarregando modelo de embeddings...")
    embedder = SentenceTransformer(EMBED_MODEL)

    textos = [c["texto"] for c in todos_chunks]
    print(f"Gerando embeddings para {len(textos)} chunks...")
    vetores = embedder.encode(
        textos,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)

    # ── 3. Constrói índice FAISS ──────────────────────────────────────────────
    print("\nConstruindo índice FAISS...")
    dim   = vetores.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (equivale a cosine com vetores normalizados)
    index.add(vetores)

    # ── 4. Salva índice e chunks ──────────────────────────────────────────────
    faiss.write_index(index, str(INDEX_DIR / "nrs.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(todos_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Índice salvo em {INDEX_DIR}/")
    print(f"   nrs.index  — {index.ntotal} vetores")
    print(f"   chunks.json — {len(todos_chunks)} chunks")

    # ── 5. Resumo por NR ──────────────────────────────────────────────────────
    print("\nResumo por NR:")
    from collections import Counter
    contagem = Counter(c["nr"] for c in todos_chunks)
    for nr in sorted(contagem):
        print(f"  {nr}: {contagem[nr]} chunks")


if __name__ == "__main__":
    main()
