"""
rag_engine.py — MPA Móveis
Motor RAG com busca única por prioridade de NR,
análise individual por produto e comparação entre dois produtos.
"""

import json
from pathlib import Path

import faiss
from google import genai
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR     = Path("data/index")
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL  = "gemini-2.5-flash"

# Ordem de prioridade fixa das NRs para o catálogo MPA
NR_PRIORIDADE = ["NR-24", "NR-18", "NR-31", "NR-17", "NR-21", "NR-01"]
CHUNKS_POR_NR = 3   # chunks por NR no contexto final
TOP_K_BUSCA   = 750 # cobre 100% do índice — custo zero em tokens e tempo


class RAGEngine:
    def __init__(self, gemini_api_key: str):
        self.client   = genai.Client(api_key=gemini_api_key)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index    = faiss.read_index(str(INDEX_DIR / "nrs.index"))
        with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # ── Busca semântica — encoding único, filtro por NR ──────────────────────
    def _buscar_por_prioridade(self, consulta: str) -> list[dict]:
        """
        Encoda a consulta UMA única vez, faz UMA busca ampla no FAISS
        e distribui os resultados por NR respeitando a ordem de prioridade.
        """
        # 1. Encoding único
        vetor = self.embedder.encode(
            [consulta],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # 2. Busca ampla — candidatos de todas as NRs de uma vez
        _, indices = self.index.search(vetor, TOP_K_BUSCA)
        candidatos = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

        # 3. Agrupa por NR preservando a ordem de relevância do FAISS
        por_nr: dict[str, list[dict]] = {nr: [] for nr in NR_PRIORIDADE}
        for c in candidatos:
            nr = c.get("nr")
            if nr in por_nr and len(por_nr[nr]) < CHUNKS_POR_NR:
                por_nr[nr].append(c)

        # 4. Combina na ordem de prioridade
        return [c for nr in NR_PRIORIDADE for c in por_nr[nr]]

    def _montar_contexto(self, chunks: list[dict]) -> str:
        return "\n\n---\n\n".join(f"[{c['nr']}]\n{c['texto']}" for c in chunks)


    def _gerar_com_retry(self, prompt: str, tentativas: int = 3) -> object:
        """Tenta gerar conteúdo com retry automático para erros 503 (sobrecarga)."""
        import time
        from google.genai import errors as genai_errors

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
                espera = tentativa * 5  # 5s, 10s, 15s
                time.sleep(espera)

    # ── Prompt: ficha individual por produto ────────────────────────────────
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

Os trechos das NRs abaixo estão organizados em ordem de prioridade para análise:
1ª NR-24 (Condições Sanitárias e de Conforto) — principal para alojamentos, vestiários e áreas de vivência
2ª NR-18 (Construção Civil) — complementa NR-24 para canteiros de obra
3ª NR-31 (Agronegócio) — complementa NR-24 para atividades rurais
4ª NR-17 (Ergonomia) — aplicável a postos de trabalho, mesas e cadeiras
5ª NR-21 (Trabalho a Céu Aberto) — aplicável a ambientes externos
6ª NR-01 (Disposições Gerais) — base legal geral

Trechos recuperados:
{contexto}

---
PRODUTO: {nome}
ESPECIFICAÇÕES TÉCNICAS: {specs_info}
LOCAL DE UTILIZAÇÃO: {local}{complemento_txt}
---

Gere uma ficha de conformidade legal para este produto neste ambiente específico.

REGRAS OBRIGATÓRIAS:
1. Analise SEMPRE começando pela NR-24. Se ela se aplicar, cite seus subitens. Só avance para as demais se necessário.
2. Cite SEMPRE o número do subitem da NR (ex: subitem 24.6.3, item 18.4.2.1). Nunca cite uma norma sem o subitem específico.
3. Se as especificações técnicas ou o tipo de espaço (vestiário, refeitório, alojamento, etc.) estiverem AUSENTES ou INSUFICIENTES para uma análise precisa, NÃO invente. Liste as perguntas exatas que precisam ser respondidas.
4. Seja objetivo e consistente. Não adicione ressalvas genéricas se as especificações já atendem ao que a NR exige.
5. Se o produto claramente não se aplica a nenhuma das NRs consultadas, informe diretamente.
6. ATRIBUTOS TÉCNICOS EXIGÍVEIS: verifique nos trechos das NRs quais atributos técnicos a norma exige para este tipo de produto (ex: densidade, dimensões mínimas, material, resistência, certificação). Se algum atributo exigível pela NR NÃO foi informado nas especificações do produto, trate-o como informação faltante e inclua a pergunta correspondente no campo de informações ao vendedor — mesmo que outros atributos estejam corretos e a análise parcial já seja possível.

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

    # ── Prompt: comparação entre dois produtos ───────────────────────────────
    def _prompt_comparacao(
        self,
        produto_a: dict,
        produto_b: dict,
        local: str,
        contexto: str,
    ) -> str:
        return f"""Você é especialista em NRs do Ministério do Trabalho e Emprego.

Ordem de prioridade das NRs:
1ª NR-24 · 2ª NR-18 · 3ª NR-31 · 4ª NR-17 · 5ª NR-21 · 6ª NR-01

Trechos recuperados:
{contexto}

---
PRODUTO A: {produto_a['nome']}
Especificações A: {produto_a['specs'] or 'Não informadas'}

PRODUTO B: {produto_b['nome']}
Especificações B: {produto_b['specs'] or 'Não informadas'}

LOCAL: {local}
---

Compare os dois produtos sob o ponto de vista da conformidade legal com as NRs.
Comece sempre pela NR-24. Cite sempre o subitem específico que embasa cada afirmação.
Seja objetivo — não adicione ressalvas genéricas se as especificações já atendem ao que a NR exige.

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

    # ── Análise de produto único ─────────────────────────────────────────────
    def analisar_produtos(
        self,
        produtos: list[dict],
        local: str,
        complementos: dict | None = None,
    ) -> list[dict]:
        complementos = complementos or {}
        resultados   = []

        for i, produto in enumerate(produtos):
            # Enriquece a query com termos de contexto para ajudar a busca semântica
            contexto_busca = "alojamento dormitório vestiário refeitório área de vivência trabalhadores"
            consulta = f"{produto['nome']} {produto['specs']} {local} {contexto_busca} conformidade NR"
            chunks   = self._buscar_por_prioridade(consulta)
            contexto = self._montar_contexto(chunks)
            prompt   = self._prompt_produto(
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

    # ── Comparação de dois produtos ──────────────────────────────────────────
    def comparar_produtos(
        self,
        produto_a: dict,
        produto_b: dict,
        local: str,
    ) -> str:
        consulta = (
            f"{produto_a['nome']} {produto_b['nome']} "
            f"{local} comparação conformidade NR"
        )
        chunks   = self._buscar_por_prioridade(consulta)
        contexto = self._montar_contexto(chunks)
        prompt   = self._prompt_comparacao(
            produto_a=produto_a,
            produto_b=produto_b,
            local=local,
            contexto=contexto,
        )

        resposta = self._gerar_com_retry(prompt)
        return resposta.text

    # ── Consultoria livre ────────────────────────────────────────────────────
    def consultar(
        self,
        pergunta: str,
        historico: list[dict] | None = None,
    ) -> str:
        """
        Responde perguntas livres sobre NRs, dimensionamento e conformidade.
        Usa o índice FAISS como base de conhecimento e mantém histórico da conversa.
        historico: lista de {"role": "user"|"assistant", "content": str}
        """
        historico = historico or []

        # Busca chunks relevantes para a pergunta
        chunks   = self._buscar_por_prioridade(pergunta)
        contexto = self._montar_contexto(chunks)

        # Monta histórico formatado
        historico_txt = ""
        if historico:
            for msg in historico[-6:]:  # últimas 6 mensagens para não estourar contexto
                papel = "Vendedor" if msg["role"] == "user" else "Assistente"
                historico_txt += f"{papel}: {msg['content']}\n\n"

        prompt = f"""Você é um consultor especialista em NRs do Ministério do Trabalho, auxiliando vendedores da MPA Móveis a entender legislação trabalhista, dimensionar espaços e identificar o que é obrigatório em alojamentos, vestiários e refeitórios.

Ordem de prioridade das NRs:
1ª NR-24 (Condições Sanitárias e de Conforto) — principal para alojamentos e áreas de vivência
2ª NR-18 (Construção Civil)
3ª NR-31 (Agronegócio)
4ª NR-17 (Ergonomia)
5ª NR-21 (Trabalho a Céu Aberto)
6ª NR-01 (Disposições Gerais)

Trechos das NRs recuperados para esta consulta:
{contexto}

---
{f"Histórico da conversa:{chr(10)}{historico_txt}" if historico_txt else ""}
Pergunta atual: {pergunta}
---

REGRAS:
1. Responda de forma clara e objetiva, em linguagem acessível para vendedores.
2. Sempre cite o subitem específico da NR que embasa cada afirmação (ex: NR-24, subitem 24.7.3 g).
3. Se a pergunta envolver cálculo de dimensionamento de espaço, SEMPRE use a área mínima definida pela NR-24 subitem 24.7.3 g): 3,00m² por cama simples e 4,50m² por beliche, já incluídas circulação e armário. Apresente o cálculo passo a passo e cite o subitem.
4. Se envolver lista de itens obrigatórios, liste todos com a NR correspondente.
5. Se não houver informação suficiente nos trechos das NRs para responder com precisão, diga claramente e indique qual NR o vendedor deve consultar.
6. Ao final, se relevante, sugira produtos MPA que atendam às exigências mencionadas.
7. ALTURA ENTRE VÃOS DE BELICHE: A NR-24 vigente NÃO estabelece medida mínima em centímetros para o vão entre camas. O subitem 24.7.3 a) exige apenas espaçamentos que permitam ao trabalhador movimentação com segurança. As medidas de 0,90m a 1,20m são referências de boas práticas do mercado baseadas na NR-17 (ergonomia), não são obrigações legais com subitem específico. NUNCA afirme que uma medida de vão está dentro das normas sem deixar claro que a norma vigente usa critério de desempenho, não medida em centímetros. Em fiscalizações, o fiscal avalia ergonomia e conforto, não uma medida exata."""

        resposta = self._gerar_com_retry(prompt)
        return resposta.text