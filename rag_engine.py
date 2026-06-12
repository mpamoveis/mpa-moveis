"""
rag_engine.py — MPA Móveis
Motor RAG híbrido:
  - NR-24, NR-18, NR-17, NR-31, NR-21 → texto COMPLETO fixo no contexto
    (elimina alucinação de subitens — o Gemini lê a norma real, não chunks parciais)
  - NR-01 → FAISS (base legal geral, raramente consultada por subitem)

Ordem de prioridade das NRs no contexto:
  1ª NR-24 (principal — condições sanitárias e alojamento)
  2ª NR-18 (construção civil)
  3ª NR-31 (agronegócio)
  4ª NR-17 (ergonomia)
  5ª NR-21 (trabalho a céu aberto)
  6ª NR-01 (disposições gerais — via FAISS)
"""

import json
import time
from pathlib import Path

import faiss
from google import genai
from google.genai import errors as genai_errors
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

INDEX_DIR    = Path("data/index")
PDF_DIR      = Path("data/pdfs")
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# NRs carregadas como texto completo (ordem de prioridade no contexto)
NR_TEXTO_COMPLETO = [
    ("NR-24", "nr-24-atualizada-2022.pdf"),
    ("NR-18", "nr-18-moveis.txt"),
    ("NR-31", "nr-31-moveis.txt"),
    ("NR-17", "nr-17-moveis.txt"),
    ("NR-21", "nr-21.pdf"),
]

# NR-01 fica no FAISS
CHUNKS_NR01 = 5      # chunks da NR-01 recuperados por consulta
TOP_K_FAISS = 200    # candidatos buscados no FAISS (ajuste após build_index)


def _extrair_texto(caminho: Path) -> str:
    """Extrai texto de PDF ou TXT."""
    if caminho.suffix.lower() == ".pdf":
        reader = PdfReader(str(caminho))
        return "\n".join(
            p.extract_text() or "" for p in reader.pages
        )
    return caminho.read_text(encoding="utf-8")


class RAGEngine:
    def __init__(self, gemini_api_key: str):
        self.client   = genai.Client(api_key=gemini_api_key)
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # ── Carrega NRs completas em memória uma única vez ────────────────────
        self.nrs_completas: dict[str, str] = {}
        for label, nome_arquivo in NR_TEXTO_COMPLETO:
            caminho = PDF_DIR / nome_arquivo
            if caminho.exists():
                self.nrs_completas[label] = _extrair_texto(caminho)
            else:
                print(f"[RAGEngine] ⚠️  Arquivo não encontrado: {nome_arquivo}")

        # ── Carrega índice FAISS (NR-01) ──────────────────────────────────────
        try:
            self.index = faiss.read_index(str(INDEX_DIR / "nrs.index"))
            with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[RAGEngine] FAISS carregado: {self.index.ntotal} vetores (NR-01)")
        except Exception as e:
            print(f"[RAGEngine] ⚠️  FAISS não carregado: {e}")
            self.index  = None
            self.chunks = []

        print(f"[RAGEngine] NRs completas carregadas: {list(self.nrs_completas.keys())}")

    # ── Monta contexto completo para o Gemini ─────────────────────────────────
    def _montar_contexto(self, consulta: str) -> str:
        """
        Contexto = NRs completas (em ordem de prioridade) + chunks NR-01 via FAISS.
        As NRs completas vêm PRIMEIRO — o Gemini lê o texto real antes de qualquer
        chunk parcial, eliminando a causa raiz de alucinação de subitens.
        """
        partes = []

        # 1. NRs completas — texto real, sem chunking, sem perda de subitens
        for label, texto in self.nrs_completas.items():
            partes.append(
                f"{'='*60}\n"
                f"[{label} — TEXTO COMPLETO]\n"
                f"{'='*60}\n"
                f"{texto.strip()}"
            )

        # 2. NR-01 via FAISS — apenas se índice disponível
        if self.index is not None and self.chunks:
            vetor = self.embedder.encode(
                [consulta],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

            k = min(TOP_K_FAISS, self.index.ntotal)
            _, indices = self.index.search(vetor, k)

            chunks_nr01 = []
            for i in indices[0]:
                if i < len(self.chunks):
                    c = self.chunks[i]
                    if c.get("nr") == "NR-01" and len(chunks_nr01) < CHUNKS_NR01:
                        chunks_nr01.append(c["texto"])

            if chunks_nr01:
                partes.append(
                    f"{'='*60}\n"
                    f"[NR-01 — trechos mais relevantes para esta consulta]\n"
                    f"{'='*60}\n"
                    + "\n\n---\n\n".join(chunks_nr01)
                )

        return "\n\n" + "\n\n".join(partes)

    # ── Retry automático para erros 503 ──────────────────────────────────────
    def _gerar_com_retry(self, prompt: str, tentativas: int = 3) -> object:
        for tentativa in range(1, tentativas + 1):
            try:
                return self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={"temperature": 0, "thinking_config": {"thinking_budget": 0}},
                )
            except genai_errors.ServerError as e:
                if tentativa == tentativas:
                    raise
                time.sleep(tentativa * 5)

    # ── Prompt: ficha individual por produto ──────────────────────────────────
    def _prompt_produto(
        self,
        nome: str,
        specs: str,
        local: str,
        contexto: str,
        complemento: str = "",
    ) -> str:
        specs_info      = specs if specs else "Não informadas"
        complemento_txt = (
            f"\nINFORMAÇÕES COMPLEMENTARES FORNECIDAS PELO CLIENTE:\n{complemento}"
            if complemento else ""
        )

        return f"""Você é especialista em legislação trabalhista brasileira e NRs do Ministério do Trabalho.

ATENÇÃO — REGRA ANTI-ALUCINAÇÃO (PRIORIDADE MÁXIMA):
Os textos COMPLETOS das NRs estão abaixo. Você DEVE:
1. Citar APENAS subitens que apareçam LITERALMENTE no texto das NRs abaixo
2. Copiar citações PALAVRA POR PALAVRA — nunca parafrasear como se fosse citação
3. NUNCA inventar ou completar subitens que não estejam no texto fornecido
4. Se não encontrar o subitem exato, escreva "conforme NR-24" sem número específico
5. Especificações numéricas (medidas, proporções) APENAS se aparecerem no texto abaixo

As NRs estão em ordem de prioridade — comece sempre pela NR-24:

{contexto}

---
PRODUTO: {nome}
ESPECIFICAÇÕES TÉCNICAS: {specs_info}
LOCAL DE UTILIZAÇÃO: {local}{complemento_txt}
---

Gere uma ficha de conformidade legal para este produto neste ambiente específico.

REGRAS DE ANÁLISE:
1. Analise SEMPRE começando pela NR-24. Se ela se aplicar, cite seus subitens. Só avance para as demais se necessário.
2. Cite SEMPRE o número do subitem da NR (ex: subitem 24.6.3, item 18.4.2.1) — mas APENAS se o subitem estiver no texto acima.
3. Se as especificações técnicas ou o tipo de espaço estiverem AUSENTES ou INSUFICIENTES, NÃO invente. Liste as perguntas exatas que precisam ser respondidas.
4. Seja objetivo e consistente. Não adicione ressalvas genéricas se as especificações já atendem ao que a NR exige.
5. Se o produto claramente não se aplica a nenhuma das NRs consultadas, informe diretamente.
6. ATRIBUTOS TÉCNICOS EXIGÍVEIS: verifique nos textos das NRs quais atributos a norma exige para este tipo de produto. Se algum atributo exigível NÃO foi informado nas especificações, trate como informação faltante.
7. ALTURA ENTRE VÃOS DE BELICHE: a NR-24 vigente NÃO estabelece medida mínima em centímetros — o subitem 24.7.3 a) exige apenas espaçamentos que permitam movimentação com segurança. Nunca afirme que uma medida está "dentro das normas" sem deixar claro que a norma usa critério de desempenho, não medida exata.

Responda EXATAMENTE neste formato:

**{nome}** — Conformidade Legal

**Norma(s) Principal(is):**
[Liste as NRs e subitens aplicáveis, começando pela NR-24 se aplicável]

**Situação de conformidade:**
[CONFORME / NÃO CONFORME / CONDICIONAL / INFORMAÇÕES INSUFICIENTES]

**Restrição de uso e justificativa:**
[Explique o que a norma exige, citando o subitem. Se for CONFORME, explique por que atende.]

**Locais PERMITIDOS:**
[Liste com justificativa de NR]

**Locais com RESTRIÇÕES ou PROIBIÇÕES:**
[Liste com justificativa de NR. Se não houver restrições, informe.]

**📋 Informações que o vendedor deve solicitar ao cliente ou fornecedor:**
[Se houver dados insuficientes, liste as perguntas exatas numeradas. Se não precisar de mais informações, escreva: Nenhuma — análise completa.]

**📲 Mensagem para WhatsApp:**
[Texto pronto, em linguagem acessível, que o vendedor copia e envia ao cliente. Máximo 4 parágrafos. Deve mencionar conformidade com NRs, risco de multa/embargo se não atender, e como o produto resolve isso.]"""

    # ── Prompt: comparação entre dois produtos ────────────────────────────────
    def _prompt_comparacao(
        self,
        produto_a: dict,
        produto_b: dict,
        local: str,
        contexto: str,
    ) -> str:
        return f"""Você é especialista em NRs do Ministério do Trabalho e Emprego.

ATENÇÃO — REGRA ANTI-ALUCINAÇÃO (PRIORIDADE MÁXIMA):
Cite APENAS subitens que apareçam LITERALMENTE nos textos das NRs abaixo.
NUNCA invente subitens. Em dúvida, cite apenas "NR-24" sem número específico.

Textos completos das NRs (em ordem de prioridade):

{contexto}

---
PRODUTO A: {produto_a['nome']}
Especificações A: {produto_a['specs'] or 'Não informadas'}

PRODUTO B: {produto_b['nome']}
Especificações B: {produto_b['specs'] or 'Não informadas'}

LOCAL: {local}
---

Compare os dois produtos sob o ponto de vista da conformidade legal com as NRs.
Comece sempre pela NR-24. Cite sempre o subitem específico que embasa cada afirmação — apenas subitens presentes no texto acima.

Responda EXATAMENTE neste formato:

## Comparativo de conformidade legal

**Ambiente analisado:** {local}

---

### Produto A — {produto_a['nome']}
**Conformidade:** [CONFORME / NÃO CONFORME / CONDICIONAL]
**Normas aplicáveis:** [NR e subitens — começando pela NR-24 se aplicável]
**Pontos fortes legais:** [O que este produto garante em termos de NR]
**Pontos de atenção:** [Limitações ou exigências adicionais]

---

### Produto B — {produto_b['nome']}
**Conformidade:** [CONFORME / NÃO CONFORME / CONDICIONAL]
**Normas aplicáveis:** [NR e subitens — começando pela NR-24 se aplicável]
**Pontos fortes legais:** [O que este produto garante em termos de NR]
**Pontos de atenção:** [Limitações ou exigências adicionais]

---

### 🏆 Recomendação final
**Melhor opção para este ambiente:** [Produto A ou B — justifique com NR]
**Argumento de venda:** [Por que a escolha recomendada protege melhor o cliente de multas e autuações do MTE]

---

### 📲 Mensagem para WhatsApp
[Texto pronto para o vendedor enviar ao cliente comparando os dois produtos.
Linguagem acessível, máximo 4 parágrafos, mencione o risco de autuação e como a escolha certa evita problemas.]"""

    # ── Análise de produto único ──────────────────────────────────────────────
    def analisar_produtos(
        self,
        produtos: list[dict],
        local: str,
        complementos: dict | None = None,
    ) -> list[dict]:
        complementos = complementos or {}
        resultados   = []

        # Monta contexto UMA vez — reutiliza para todos os produtos da sessão
        consulta_base = f"conformidade NR alojamento dormitório vestiário refeitório área de vivência {local}"
        contexto = self._montar_contexto(consulta_base)

        for i, produto in enumerate(produtos):
            prompt = self._prompt_produto(
                nome=produto["nome"],
                specs=produto["specs"],
                local=local,
                contexto=contexto,
                complemento=complementos.get(i, ""),
            )
            resposta = self._gerar_com_retry(prompt)
            resultados.append({
                "nome":  produto["nome"],
                "ficha": resposta.text,
            })

        return resultados

    # ── Comparação de dois produtos ───────────────────────────────────────────
    def comparar_produtos(
        self,
        produto_a: dict,
        produto_b: dict,
        local: str,
    ) -> str:
        consulta = f"{produto_a['nome']} {produto_b['nome']} {local} comparação conformidade NR"
        contexto = self._montar_contexto(consulta)
        prompt   = self._prompt_comparacao(
            produto_a=produto_a,
            produto_b=produto_b,
            local=local,
            contexto=contexto,
        )
        resposta = self._gerar_com_retry(prompt)
        return resposta.text

    # ── Consultoria livre ─────────────────────────────────────────────────────
    def consultar(
        self,
        pergunta: str,
        historico: list[dict] | None = None,
    ) -> str:
        historico = historico or []
        contexto  = self._montar_contexto(pergunta)

        historico_txt = ""
        if historico:
            for msg in historico[-6:]:
                papel = "Vendedor" if msg["role"] == "user" else "Assistente"
                historico_txt += f"{papel}: {msg['content']}\n\n"

        prompt = f"""Você é um consultor especialista em NRs do Ministério do Trabalho, auxiliando vendedores da MPA Móveis a entender legislação trabalhista, dimensionar espaços e identificar o que é obrigatório em alojamentos, vestiários e refeitórios.

ATENÇÃO — REGRA ANTI-ALUCINAÇÃO (PRIORIDADE MÁXIMA):
Os textos COMPLETOS das NRs estão abaixo. Você DEVE:
1. Citar APENAS subitens que apareçam LITERALMENTE nos textos abaixo
2. NUNCA inventar ou completar subitens
3. Especificações numéricas APENAS se aparecerem no texto das NRs abaixo
4. Em dúvida, cite "NR-24" sem número de subitem

Textos completos das NRs (ordem de prioridade — NR-24 primeiro):

{contexto}

---
{f"Histórico da conversa:{chr(10)}{historico_txt}" if historico_txt else ""}
Pergunta atual: {pergunta}
---

REGRAS DE RESPOSTA:
1. Responda de forma clara e objetiva, em linguagem acessível para vendedores.
2. Sempre cite o subitem específico da NR — mas APENAS se ele estiver no texto acima.
3. Se a pergunta envolver cálculo de dimensionamento, use SEMPRE NR-24 subitem 24.7.3 g): 3,00m² por cama simples e 4,50m² por beliche, já incluídas circulação e armário. Apresente o cálculo passo a passo.
4. Se envolver lista de itens obrigatórios, liste todos com a NR correspondente.
5. Se não houver informação suficiente nos textos das NRs para responder com precisão, diga claramente.
6. Ao final, se relevante, sugira produtos MPA que atendam às exigências mencionadas.
7. ALTURA ENTRE VÃOS DE BELICHE: a NR-24 vigente NÃO estabelece medida mínima em centímetros para o vão entre camas. O subitem 24.7.3 a) exige apenas espaçamentos que permitam movimentação com segurança. As medidas de 0,90m a 1,20m são referências de mercado, não obrigações legais com subitem específico. NUNCA afirme que uma medida está dentro das normas sem deixar claro que a norma usa critério de desempenho."""

        resposta = self._gerar_com_retry(prompt)
        return resposta.text
