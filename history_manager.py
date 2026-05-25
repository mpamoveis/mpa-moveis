"""
history_manager.py — MPA Móveis
Gerencia o histórico de análises no Google Sheets.
Registra: usuário, data, hora, produto, local e resultado.
"""

import json
from datetime import datetime
import pytz

import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CABECALHO = [
    "Data",
    "Hora",
    "Usuário",
    "Produto",
    "Local",
    "Status",
    "Resultado completo",
]


class HistoryManager:
    def __init__(self, creds_json: str, sheet_id: str):
        """
        creds_json: conteúdo do arquivo JSON da conta de serviço (string)
        sheet_id: ID da planilha do Google Sheets
        """
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        self.client   = gspread.authorize(creds)
        self.sheet_id = sheet_id
        self._garantir_cabecalho()

    def _abrir_planilha(self):
        return self.client.open_by_key(self.sheet_id).sheet1

    def _garantir_cabecalho(self):
        """Cria o cabeçalho se a planilha estiver vazia."""
        try:
            sheet = self._abrir_planilha()
            if not sheet.row_values(1):
                sheet.append_row(CABECALHO)
        except Exception:
            pass

    def _detectar_status(self, ficha: str) -> str:
        ficha_upper = ficha.upper()
        if "INFORMAÇÕES INSUFICIENTES" in ficha_upper:
            return "INFORMAÇÕES INSUFICIENTES"
        if "NÃO CONFORME" in ficha_upper or "NAO CONFORME" in ficha_upper:
            return "NÃO CONFORME"
        if "CONDICIONAL" in ficha_upper:
            return "CONDICIONAL"
        if "CONFORME" in ficha_upper:
            return "CONFORME"
        return "INDEFINIDO"

    def salvar(
        self,
        usuario: str,
        produtos: list[dict],
        local: str,
        resultado: list[dict],
        tipo_espaco: str = "—",
    ) -> bool:
        """
        Salva cada produto analisado como uma linha na planilha.
        Retorna True se salvo com sucesso.
        """
        try:
            sheet = self._abrir_planilha()
            fuso = pytz.timezone('America/Sao_Paulo')
        agora = datetime.now(fuso)
            data  = agora.strftime("%d/%m/%Y")
            hora  = agora.strftime("%H:%M:%S")

            for res in resultado:
                status = self._detectar_status(res["ficha"])
                # Limita o resultado a 5000 caracteres para não exceder o limite do Sheets
                resultado_txt = res["ficha"][:5000]

                sheet.append_row([
                    data,
                    hora,
                    usuario,
                    res["nome"],
                    local,
                    status,
                    resultado_txt,
                ])

            return True
        except Exception as e:
            print(f"Erro ao salvar histórico: {e}")
            return False

    def buscar(self, termo: str = "") -> list[dict]:
        """
        Busca registros no histórico.
        Se termo for vazio, retorna os 20 mais recentes.
        """
        try:
            sheet    = self._abrir_planilha()
            registros = sheet.get_all_records()

            if not registros:
                return []

            # Filtra por termo se informado
            if termo.strip():
                termo_lower = termo.strip().lower()
                registros = [
                    r for r in registros
                    if termo_lower in str(r.get("Produto", "")).lower()
                    or termo_lower in str(r.get("Local", "")).lower()
                    or termo_lower in str(r.get("Usuário", "")).lower()
                ]

            # Ordena do mais recente para o mais antigo e limita a 50
            registros = list(reversed(registros))[:50]

            return [
                {
                    "data":            f"{r.get('Data', '')} {r.get('Hora', '')}",
                    "usuario":         r.get("Usuário", "—"),
                    "produtos_resumo": r.get("Produto", "—"),
                    "local":           r.get("Local", "—"),
                    "status":          r.get("Status", "—"),
                    "resultado_texto": r.get("Resultado completo", ""),
                }
                for r in registros
            ]
        except Exception as e:
            print(f"Erro ao buscar histórico: {e}")
            return []
