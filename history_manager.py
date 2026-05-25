"""
history_manager.py — MPA Móveis
Salva e busca análises no Google Sheets.

Configuração necessária:
- GOOGLE_SHEETS_CREDS: conteúdo do JSON da service account (string no .env)
- GOOGLE_SHEET_ID: ID da planilha (na URL entre /d/ e /edit)

A planilha precisa ter os cabeçalhos na linha 1:
data | usuario | local | tipo_espaco | produtos_resumo | resultado_texto
"""

import json
import os
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLUNAS = ["data", "usuario", "local", "tipo_espaco", "produtos_resumo", "resultado_texto"]


class HistoryManager:
    def __init__(self, creds_json: str, sheet_id: str):
        """
        creds_json: conteúdo completo do arquivo JSON da service account
        sheet_id:   ID da planilha Google Sheets
        """
        creds_dict = json.loads(creds_json)
        creds      = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client     = gspread.authorize(creds)
        planilha   = client.open_by_key(sheet_id)

        # Usa a primeira aba ou cria "Histórico" se não existir
        try:
            self.aba = planilha.worksheet("Histórico")
        except gspread.WorksheetNotFound:
            self.aba = planilha.add_worksheet(title="Histórico", rows=1000, cols=10)
            self.aba.append_row(COLUNAS)

    # ── Salva nova análise ───────────────────────────────────────────────────
    def salvar(
        self,
        usuario: str,
        produtos: list[dict],
        local: str,
        tipo_espaco: str,
        resultado: list[dict],
    ):
        produtos_resumo = " | ".join(p["nome"] for p in produtos)
        resultado_texto = "\n\n".join(
            f"=== {r['nome']} ===\n{r['ficha']}" for r in resultado
        )

        self.aba.append_row([
            datetime.now().strftime("%d/%m/%Y %H:%M"),
            usuario,
            local,
            tipo_espaco,
            produtos_resumo,
            resultado_texto,
        ])

    # ── Busca por nome de produto ────────────────────────────────────────────
    def buscar(self, termo: str = "") -> list[dict]:
        try:
            todas = self.aba.get_all_records()
        except Exception:
            return []

        if not termo.strip():
            # Retorna os 20 mais recentes quando não há termo
            registros = todas[-20:]
            registros.reverse()
            return registros

        termo_lower = termo.lower()
        encontrados = [
            r for r in todas
            if termo_lower in r.get("produtos_resumo", "").lower()
        ]
        encontrados.reverse()  # mais recente primeiro
        return encontrados
