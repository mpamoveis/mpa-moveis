"""
build_index.py — MPA Móveis
Reconstrói o índice FAISS com APENAS a NR-01 (base legal geral).

As demais NRs são carregadas como texto completo diretamente no contexto do Gemini:
  - nr-17-moveis.txt  (8.7k chars)  — trecho ergonomia relevante para móveis
  - nr-18-moveis.txt  (1.9k chars)  — trecho construção civil
  - nr-31-moveis.txt  (8.6k chars)  — trecho agronegócio
  - nr-21.pdf         (2.5k chars)  — trabalho a céu aberto (completa)
  - nr-24-atualizada-2022.pdf (31k) — condições sanitárias (completa — a mais importante)

Apenas a NR-01 (53k chars) fica no FAISS por ser base legal geral
raramente consultada por subitem específico.

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

# ── Configuração ──────────────────────────────────────────────────────────────
PDF_DIR     = Path("data/pdfs")
INDEX_DIR   = Path("data/index")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_MAX = 2000
CHUNK_MIN = 100
OVERLAP   = 400

# Apenas NR-01 vai para o FAISS
NR_FAISS = {
    "nr-01": "NR-01",
}

# NRs que ficam como texto completo (não são indexadas no FAISS)
NR_TEXTO_COMPLETO = [
    "nr-17-moveis.txt",
    "nr-18-moveis.txt",
    "nr-24-atualizada-2022.pdf",
    "nr-31-moveis.txt",
    "nr-21.pdf",
]


# ── Extração de texto ─────────────────────────────────────────────────────────
def extrair_texto_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    paginas = []
    for page in reader.pages:
        texto = page.extract_text()
        if texto:
            paginas.append(texto)
    return "\n".join(paginas)


def extrair_texto_txt(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8")


def extrair_texto(arquivo: Path) -> str:
    if arquivo.suffix.lower() == ".pdf":
        return extrair_texto_pdf(arquivo)
    elif arquivo.suffix.lower() == ".txt":
        return extrair_texto_txt(arquivo)
    return ""


# ── Chunking com overlap ──────────────────────────────────────────────────────
def chunk_texto(texto: str, nr_label: str) -> list[dict]:
    chunks = []
    padrao_subitem = re.compile(
        r'(?=\n\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})*(?:\.\d{1,2})?\s)',
        re.MULTILINE
    )
    segmentos = padrao_subitem.split(texto)
    buffer = ""

    for seg in segmentos:
        seg = seg.strip()
        if not seg:
            continue
        if len(buffer) + len(seg) <= CHUNK_MAX:
            buffer += "\n" + seg if buffer else seg
        else:
            if len(buffer) >= CHUNK_MIN:
                chunks.append({"nr": nr_label, "texto": buffer.strip()})
            overlap_texto = buffer[-OVERLAP:] if len(buffer) > OVERLAP else buffer
            buffer = overlap_texto + "\n" + seg if overlap_texto else seg

    if buffer.strip() and len(buffer.strip()) >= CHUNK_MIN:
        chunks.append({"nr": nr_label, "texto": buffer.strip()})

    return chunks


# ── Detecta se arquivo é NR-01 ────────────────────────────────────────────────
def detectar_nr_faiss(arquivo: Path) -> str | None:
    nome = arquivo.stem.lower()
    for chave, label in NR_FAISS.items():
        if chave in nome:
            return label
    return None


# ── Valida arquivos de texto completo ─────────────────────────────────────────
def validar_texto_completo():
    print("\nValidando NRs de texto completo:")
    for nome_arquivo in NR_TEXTO_COMPLETO:
        caminho = PDF_DIR / nome_arquivo
        if not caminho.exists():
            print(f"  ⚠️  NÃO ENCONTRADO: {nome_arquivo}")
            continue
        texto = extrair_texto(caminho)
        print(f"  ✅ {nome_arquivo}: {len(texto):,} chars ({len(texto)//4:,} tokens aprox.)")


# ── Pipeline principal ────────────────────────────────────────────────────────
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Valida NRs de texto completo ─────────────────────────────────────────
    validar_texto_completo()

    # ── Encontra arquivos para FAISS (apenas NR-01) ───────────────────────────
    arquivos_faiss = []
    for arquivo in sorted(PDF_DIR.iterdir()):
        if arquivo.suffix.lower() not in (".pdf", ".txt"):
            continue
        # Pula NRs de texto completo
        if arquivo.name in NR_TEXTO_COMPLETO:
            continue
        nr_label = detectar_nr_faiss(arquivo)
        if nr_label:
            arquivos_faiss.append((arquivo, nr_label))

    if not arquivos_faiss:
        print("\n⚠️  Nenhum arquivo encontrado para indexar no FAISS.")
        print("   Verifique se nr-01-atualizada-26.05.26.pdf está em data/pdfs/")
        return

    print(f"\nArquivos para FAISS: {len(arquivos_faiss)}")
    for arq, label in arquivos_faiss:
        print(f"  - {arq.name} → {label}")

    # ── Extrai e chunka ───────────────────────────────────────────────────────
    todos_chunks: list[dict] = []

    for arquivo, nr_label in arquivos_faiss:
        print(f"\nProcessando {arquivo.name} → {nr_label}")
        texto  = extrair_texto(arquivo)
        chunks = chunk_texto(texto, nr_label)
        todos_chunks.extend(chunks)
        print(f"  Chunks gerados: {len(chunks)}")

    if not todos_chunks:
        print("\n⚠️  Nenhum chunk gerado. Verifique os arquivos.")
        return

    print(f"\nTotal de chunks FAISS: {len(todos_chunks)}")

    # ── Embeddings ────────────────────────────────────────────────────────────
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

    # ── Índice FAISS ──────────────────────────────────────────────────────────
    print("\nConstruindo índice FAISS...")
    dim   = vetores.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vetores)

    # ── Salva ─────────────────────────────────────────────────────────────────
    faiss.write_index(index, str(INDEX_DIR / "nrs.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(todos_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Índice FAISS salvo em {INDEX_DIR}/")
    print(f"   nrs.index   — {index.ntotal} vetores (NR-01)")
    print(f"   chunks.json — {len(todos_chunks)} chunks")

    print("\nResumo por NR no FAISS:")
    from collections import Counter
    contagem = Counter(c["nr"] for c in todos_chunks)
    for nr in sorted(contagem):
        print(f"  {nr}: {contagem[nr]} chunks")

    print("\n📋 NRs carregadas como texto completo (sem FAISS):")
    for nome in NR_TEXTO_COMPLETO:
        caminho = PDF_DIR / nome
        if caminho.exists():
            texto = extrair_texto(caminho)
            print(f"  {nome}: {len(texto):,} chars")

    print("\n✅ build_index concluído!")
    print("   Atualize TOP_K_BUSCA no rag_engine.py se necessário.")
    print(f"   Valor sugerido: TOP_K_BUSCA = {index.ntotal} (cobre 100% do índice NR-01)")


if __name__ == "__main__":
    main()
