# MPA Móveis — Copiloto de Vendas

Ferramenta interna com login, análise de conformidade com NRs,
comparação de produtos e histórico em Google Sheets.

---

## Estrutura

```
mpa-moveis/
├── app.py                  # Interface principal (login + 3 abas)
├── rag_engine.py           # Motor RAG — análise e comparação por produto
├── history_manager.py      # Integração Google Sheets
├── requirements.txt
├── .env.example            # Modelo de configuração
├── .gitignore
├── data/
│   ├── pdfs/               # Coloque aqui os 7 PDFs das NRs
│   └── index/              # Gerado pelo build_index.py
│       ├── nrs.index
│       └── chunks.json
└── scripts/
    └── build_index.py
```

---

## Configuração inicial (passo a passo)

### 1. Clone e instale dependências
```bash
git clone https://github.com/seu-usuario/mpa-moveis.git
cd mpa-moveis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure o arquivo .env
```bash
cp .env.example .env
# Abra o .env e preencha as variáveis
```

### 3. Coloque os PDFs em data/pdfs/
Os nomes dos arquivos precisam conter o número da NR:
- `nr-01-atualizada.pdf`
- `nr-17-ergonomia.pdf`
- `nr-18-construcao.pdf`
- `nr-21-trabalhoceudaberto.pdf`
- `nr-24-sanitarias.pdf`
- `nr-31-agronomia.pdf`
- (e o Anexo II da NR-17 se tiver separado)

### 4. Gere o índice FAISS (uma vez)
```bash
python scripts/build_index.py
```

### 5. Rode localmente
```bash
streamlit run app.py
```
Acesse: http://localhost:8501

---

## Configurar Google Sheets (histórico)

### Passo 1 — Criar o projeto no Google Cloud
1. Acesse https://console.cloud.google.com
2. Crie um novo projeto (ex: "mpa-copiloto")
3. Ative as APIs: **Google Sheets API** e **Google Drive API**

### Passo 2 — Criar Service Account
1. Em IAM & Admin → Service Accounts → Criar
2. Nome: `mpa-copiloto`
3. Clique em "Criar chave" → JSON → Baixe o arquivo

### Passo 3 — Criar a planilha
1. Crie uma planilha em Google Sheets
2. Na linha 1, adicione os cabeçalhos:
   `data | usuario | local | tipo_espaco | produtos_resumo | resultado_texto`
3. Copie o ID da planilha da URL (entre `/d/` e `/edit`)
4. Compartilhe a planilha com o e-mail da service account (permissão de Editor)

### Passo 4 — Configurar o .env
```
GOOGLE_SHEETS_CREDS={"type":"service_account",...}  # conteúdo do JSON em uma linha
GOOGLE_SHEET_ID=1aBcDeFg...
```

---

## Publicar no Streamlit Cloud

1. Suba o projeto para repositório **privado** no GitHub
   (certifique-se que `.env` está no `.gitignore`)

2. O arquivo `data/index/nrs.index` **precisa estar commitado** no GitHub

3. Acesse https://share.streamlit.io e conecte o repositório

4. Em **Advanced settings → Secrets**, adicione:
```toml
GEMINI_API_KEY = "sua_chave"
USERS = "admin:mpa2024,vendedor1:senha123"
GOOGLE_SHEETS_CREDS = '{"type":"service_account",...}'
GOOGLE_SHEET_ID = "1aBcDeFg..."
```

---

## Gerenciar usuários

Adicione ou remova usuários alterando a variável `USERS`:
```
USERS=admin:mpa2024,joao:senha456,maria:senha789
```
No Streamlit Cloud, atualize em Settings → Secrets e o app recarrega automaticamente.

---

## Atualizar uma NR

1. Substitua o PDF em `data/pdfs/`
2. Rode `python scripts/build_index.py`
3. Commit e push de `data/index/nrs.index` e `data/index/chunks.json`
4. O Streamlit Cloud recarrega automaticamente
