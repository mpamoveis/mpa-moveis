"""
app.py — MPA Móveis | Copiloto de Vendas
Interface principal com login, identidade visual e 3 abas.
"""

import os
import threading
import time
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine
from history_manager import HistoryManager

load_dotenv()

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MPA Móveis | Copiloto de Vendas",
    page_icon="🪑",
    layout="centered",
)

# ── Identidade visual MPA Móveis ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; color: #f0f0f0; }
    [data-testid="stSidebar"] { background-color: #111111; }
    .mpa-header { text-align: center; padding: 1.5rem 0 0.5rem; }
    .mpa-title { font-size: 28px; font-weight: 700; color: #C9A84C; letter-spacing: 2px; }
    .mpa-subtitle { font-size: 13px; color: #888; margin-top: 2px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #111; border-bottom: 1px solid #2a2a2a; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #888 !important; background-color: transparent !important; border-radius: 6px 6px 0 0 !important; padding: 8px 20px !important; font-size: 14px !important; }
    .stTabs [aria-selected="true"] { color: #C9A84C !important; border-bottom: 2px solid #C9A84C !important; background-color: #1a1a1a !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div { background-color: #242424 !important; color: #f0f0f0 !important; border: 1px solid #333 !important; border-radius: 6px !important; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #C9A84C !important; box-shadow: 0 0 0 1px #C9A84C22 !important; }
    label, .stTextInput label, .stTextArea label, .stSelectbox label { color: #aaa !important; font-size: 13px !important; }
    .stButton > button { background-color: #C9A84C !important; color: #111 !important; border: none !important; font-weight: 600 !important; border-radius: 6px !important; padding: 0.5rem 1.5rem !important; transition: background 0.15s !important; }
    .stButton > button:hover { background-color: #b8932f !important; color: #111 !important; }
    hr { border-color: #2a2a2a !important; }
    .stAlert { border-radius: 6px !important; }
    .mpa-caption { font-size: 12px; color: #666; text-align: center; margin-top: 0.5rem; }
    .produto-card { background: #242424; border: 1px solid #333; border-left: 3px solid #C9A84C; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
    .produto-label { font-size: 11px; font-weight: 700; color: #C9A84C; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.25rem; }
    .produto-nome { font-size: 18px; font-weight: 700; color: #f0f0f0; margin-bottom: 1rem; }
    .status-conforme { background: #1a3a1a; border: 1px solid #2d6a2d; border-left: 4px solid #4CAF50; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.5rem 0 1rem 0; color: #81c784; font-weight: 600; font-size: 15px; }
    .status-nao-conforme { background: #3a1a1a; border: 1px solid #6a2d2d; border-left: 4px solid #f44336; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.5rem 0 1rem 0; color: #e57373; font-weight: 600; font-size: 15px; }
    .status-condicional { background: #3a2f1a; border: 1px solid #6a5a2d; border-left: 4px solid #FF9800; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.5rem 0 1rem 0; color: #ffb74d; font-weight: 600; font-size: 15px; }
    .status-insuficiente { background: #2a2a3a; border: 1px solid #3a3a6a; border-left: 4px solid #9C27B0; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.5rem 0 1rem 0; color: #ce93d8; font-weight: 600; font-size: 15px; }
    .complemento-box { background: #1e1e2e; border: 1px solid #9C27B044; border-radius: 6px; padding: 1rem; margin: 0.5rem 0 1rem 0; }
    .chat-msg-user { background: #1e2a1e; border: 1px solid #2d4a2d; border-radius: 8px 8px 8px 0; padding: 0.75rem 1rem; margin: 0.5rem 0 0.5rem 2rem; color: #f0f0f0; font-size: 14px; }
    .chat-msg-assistant { background: #242424; border: 1px solid #333; border-left: 3px solid #C9A84C; border-radius: 8px 8px 0 8px; padding: 0.75rem 1rem; margin: 0.5rem 2rem 0.5rem 0; color: #f0f0f0; font-size: 14px; }
    .chat-role-user { font-size: 11px; color: #81c784; font-weight: 700; margin-bottom: 4px; }
    .chat-role-assistant { font-size: 11px; color: #C9A84C; font-weight: 700; margin-bottom: 4px; }
    .login-box { max-width: 360px; margin: 4rem auto 0; padding: 2rem; background: #242424; border: 1px solid #333; border-radius: 10px; }
    .login-title { text-align: center; font-size: 22px; font-weight: 700; color: #C9A84C; letter-spacing: 2px; margin-bottom: 4px; }
    .login-sub { text-align: center; font-size: 12px; color: #666; margin-bottom: 1.5rem; }
    .dica-box { background: #1e2a1e; border: 1px solid #2d4a2d; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; color: #81c784; font-size: 13px; }
    .info-vendedor { background: #2a2500; border: 1px solid #6a5c00; border-left: 4px solid #F5C400; border-radius: 6px; padding: 0.9rem 1.1rem; margin: 1rem 0; }
    .info-vendedor p, .info-vendedor li { color: #FFE066 !important; font-size: 15px !important; }
    .info-vendedor strong { color: #FFE066 !important; font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _destacar_info_vendedor(ficha: str) -> str:
    """Envolve a seção de informações ao vendedor em uma div estilizada."""
    marcador = "**\U0001f4cb Informa\u00e7\u00f5es que o vendedor deve solicitar ao cliente ou fornecedor:**"
    if marcador not in ficha:
        return ficha
    partes = ficha.split(marcador)
    if len(partes) != 2:
        return ficha
    antes, depois = partes
    linhas = depois.split("\n")
    conteudo_info = []
    resto = []
    dentro = True
    for linha in linhas:
        if dentro:
            stripped = linha.strip()
            if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                dentro = False
                resto.append(linha)
            else:
                conteudo_info.append(linha)
        else:
            resto.append(linha)
    bloco_info = "\n".join(conteudo_info)
    bloco_resto = "\n".join(resto)
    div_open = '<div class="info-vendedor">'
    div_close = "</div>"
    return antes + div_open + "\n\n" + marcador + "\n" + bloco_info + "\n" + div_close + "\n\n" + bloco_resto

def _detectar_status(ficha: str) -> str:
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


def _renderizar_status(status: str):
    badges = {
        "CONFORME":                  ("status-conforme",     "✅ CONFORME"),
        "NÃO CONFORME":              ("status-nao-conforme", "❌ NÃO CONFORME"),
        "CONDICIONAL":               ("status-condicional",  "⚠️ CONDICIONAL"),
        "INFORMAÇÕES INSUFICIENTES": ("status-insuficiente", "🔍 INFORMAÇÕES INSUFICIENTES — responda as perguntas abaixo para completar a análise"),
        "INDEFINIDO":                ("status-condicional",  "❓ STATUS INDEFINIDO"),
    }
    css, label = badges.get(status, badges["INDEFINIDO"])
    st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)


# ── Progresso com thread ─────────────────────────────────────────────────────
def _executar_com_progresso(fn, mensagens: list[str], *args, **kwargs):
    """
    Executa fn(*args, **kwargs) em uma thread e exibe barra de progresso
    com percentual real enquanto aguarda.
    mensagens: lista de textos exibidos em etapas do progresso.
    """
    resultado = [None]
    erro      = [None]

    def _thread():
        try:
            resultado[0] = fn(*args, **kwargs)
        except Exception as e:
            erro[0] = e

    t = threading.Thread(target=_thread)
    t.start()

    barra   = st.progress(0)
    status  = st.empty()
    pct     = 0
    etapa   = 0
    intervalo = 100 / max(len(mensagens), 1)

    while t.is_alive():
        etapa_atual = min(int(pct / intervalo), len(mensagens) - 1)
        status.markdown(f"⏳ {mensagens[etapa_atual]} — **{pct}%**")
        barra.progress(pct)
        time.sleep(0.4)
        if pct < 92:
            pct = min(pct + 2, 92)

    barra.progress(100)
    status.markdown("✅ **Concluído! — 100%**")
    time.sleep(0.6)
    barra.empty()
    status.empty()

    if erro[0]:
        raise erro[0]
    return resultado[0]



def verificar_login(usuario: str, senha: str) -> bool:
    usuarios_raw = os.getenv("USERS", "admin:mpa2024")
    usuarios = {}
    for par in usuarios_raw.split(","):
        partes = par.strip().split(":")
        if len(partes) == 2:
            usuarios[partes[0].strip()] = partes[1].strip()
    return usuarios.get(usuario) == senha


def tela_login():
    st.markdown("""
    <div class="login-box">
        <div class="login-title">MPA MÓVEIS</div>
        <div class="login-sub">Copiloto de Vendas — Acesso restrito</div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        usuario = st.text_input("Usuário", key="login_user")
        senha   = st.text_input("Senha", type="password", key="login_pass")
        if st.button("Entrar", use_container_width=True):
            if verificar_login(usuario, senha):
                st.session_state["autenticado"]  = True
                st.session_state["usuario_nome"] = usuario
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos.")
        st.markdown('<p class="mpa-caption">mpa.moveisparaalojamento.com.br</p>', unsafe_allow_html=True)


# ── Recursos ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando base de conhecimento das NRs...")
def carregar_engine():
    import subprocess
    from pathlib import Path
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Chave GEMINI_API_KEY não encontrada.")
        st.stop()
    if not Path("data/index/nrs.index").exists():
        Path("data/index").mkdir(parents=True, exist_ok=True)
        with st.spinner("Gerando índice das NRs pela primeira vez — pode levar alguns minutos..."):
            import sys
            subprocess.run([sys.executable, "build_index.py"], check=True)
    return RAGEngine(gemini_api_key=api_key)


@st.cache_resource
def carregar_history():
    creds_json = os.getenv("GOOGLE_SHEETS_CREDS")
    sheet_id   = os.getenv("GOOGLE_SHEET_ID")
    if not creds_json or not sheet_id:
        return None
    return HistoryManager(creds_json=creds_json, sheet_id=sheet_id)


# ── App principal ────────────────────────────────────────────────────────────
def app_principal():
    st.markdown("""
    <div class="mpa-header">
        <div class="mpa-title">MPA MÓVEIS</div>
        <div class="mpa-subtitle">moveisparaalojamento.com.br · Copiloto de Vendas</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Sair", key="logout"):
            st.session_state.clear()
            st.rerun()

    st.divider()

    engine  = carregar_engine()
    history = carregar_history()

    aba_analise, aba_comparacao, aba_consultoria, aba_historico = st.tabs([
        "🔍 Análise de Conformidade",
        "⚖️ Comparar Produtos",
        "💬 Consultoria",
        "📋 Histórico",
    ])

    # ════════════════════════════════════════════════════════════════
    # ABA 1 — ANÁLISE (produto único)
    # ════════════════════════════════════════════════════════════════
    with aba_analise:

        # ── Inicializa estado ────────────────────────────────────────
        for key, default in [
            ("resultado_analise", None),
            ("produto_analise",   None),
            ("local_analise",     ""),
            ("form_version",      0),
            ("historico_salvo",   False),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        v = st.session_state["form_version"]

        # ── Formulário ───────────────────────────────────────────────
        st.markdown("#### Local de utilização")
        local = st.selectbox("", [
            "Construção Civil",
            "Agronegócio",
            "Mineração",
            "Indústria",
            "Escritório Administrativo",
            "Call Center / Teleatendimento",
            "Infraestrutura / Obras Externas",
            "Trabalho a Céu Aberto (NR-21)",
            "Outro",
        ], key=f"local_v{v}", label_visibility="collapsed")

        detalhe = st.text_input(
            "Detalhes adicionais (opcional)",
            placeholder="Ex: 80 trabalhadores, turno noturno, zona rural...",
            key=f"detalhe_v{v}",
        )

        st.divider()
        st.markdown("#### Produto para análise")
        st.markdown(
            '<div class="dica-box">💡 Quanto mais detalhada a especificação, mais precisa e completa será a análise de conformidade.</div>',
            unsafe_allow_html=True,
        )

        nome = st.text_input(
            "Nome / modelo do produto",
            placeholder="Ex: Armário roupeiro 4 portas em aço",
            key=f"nome_v{v}",
        )
        specs = st.text_area(
            "Especificações técnicas",
            placeholder=(
                "Ex: Aço carbono 26, 4 portas com chave, divisória interna para roupa limpa e suja, "
                "pitão para cadeado, dimensões 1,98m x 1,00m x 0,40m, uso em ambiente insalubre..."
            ),
            height=140,
            key=f"specs_v{v}",
        )

        st.divider()

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            analisar = st.button("🔍 Analisar conformidade com as NRs", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Limpar", use_container_width=True, key="limpar_analise"):
                st.session_state["resultado_analise"] = None
                st.session_state["produto_analise"]   = None
                st.session_state["local_analise"]     = ""
                st.session_state["historico_salvo"]   = False
                st.session_state["form_version"]     += 1
                st.rerun()

        # ── Análise ──────────────────────────────────────────────────
        if analisar:
            if not nome.strip():
                st.warning("Preencha o nome do produto antes de analisar.")
            else:
                local_completo = f"{local} — {detalhe}" if detalhe.strip() else local
                produto = {"nome": nome.strip(), "specs": specs.strip()}

                try:
                    resultados = _executar_com_progresso(
                        engine.analisar_produtos,
                        [
                            "Preparando consulta...",
                            "Buscando trechos das NRs...",
                            "Priorizando NR-24, NR-18, NR-31...",
                            "Enviando para análise jurídica...",
                            "Gerando ficha de conformidade...",
                        ],
                        produtos=[produto],
                        local=local_completo,
                    )
                except Exception as e:
                    msg = str(e)
                    if "503" in msg or "UNAVAILABLE" in msg:
                        st.warning("⏳ O servidor do Gemini está sobrecarregado no momento. Aguarde alguns segundos e tente novamente.")
                    else:
                        st.error(f"Erro inesperado: {msg}")
                    resultados = None

                if resultados:
                    st.session_state["resultado_analise"] = resultados[0]
                    st.session_state["produto_analise"]   = produto
                    st.session_state["local_analise"]     = local_completo
                    st.session_state["historico_salvo"]   = False

        # ── Exibição do resultado ────────────────────────────────────
        res = st.session_state.get("resultado_analise")
        if res:
            status = _detectar_status(res["ficha"])
            st.divider()

            st.markdown('<div class="produto-label">Resultado da análise</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="produto-nome">{res["nome"]}</div>', unsafe_allow_html=True)
            _renderizar_status(status)

            st.markdown(_destacar_info_vendedor(res["ficha"]), unsafe_allow_html=True)

            # ── Complemento quando há informações insuficientes ──────
            if status == "INFORMAÇÕES INSUFICIENTES":
                st.markdown('<div class="complemento-box">', unsafe_allow_html=True)
                st.markdown("**📝 Responda as perguntas acima para completar a análise:**")
                complemento = st.text_area(
                    "Informações complementares",
                    height=110,
                    placeholder="Ex: O roupeiro será usado em vestiário de obra com atividade insalubre grau médio, 40 trabalhadores por turno...",
                    key=f"complemento_v{v}",
                )
                if st.button("🔄 Reanalizar com as informações complementares", use_container_width=True):
                    if complemento.strip():
                        with st.container():
                            novo = _executar_com_progresso(
                                engine.analisar_produtos,
                                [
                                    "Preparando complemento...",
                                    "Buscando NRs relevantes...",
                                    "Reanalisando com novas informações...",
                                    "Gerando ficha atualizada...",
                                ],
                                produtos=[st.session_state["produto_analise"]],
                                local=st.session_state["local_analise"],
                                complementos={0: complemento.strip()},
                            )
                        st.session_state["resultado_analise"] = novo[0]
                        st.session_state["historico_salvo"]   = False
                        st.rerun()
                    else:
                        st.warning("Preencha as informações antes de reanalizar.")
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Salva no histórico ───────────────────────────────────
            if history and not st.session_state["historico_salvo"]:
                history.salvar(
                    usuario=st.session_state.get("usuario_nome", "—"),
                    produtos=[st.session_state["produto_analise"]],
                    local=st.session_state["local_analise"],
                    tipo_espaco="—",
                    resultado=[res],
                )
                st.session_state["historico_salvo"] = True
                st.success("Análise salva no histórico.")
            elif not history:
                st.caption("Histórico não configurado — análise não salva.")

    # ════════════════════════════════════════════════════════════════
    # ABA 2 — COMPARAÇÃO
    # ════════════════════════════════════════════════════════════════
    with aba_comparacao:

        # ── Inicializa estado ────────────────────────────────────────
        for key, default in [
            ("resultado_comp",  ""),
            ("comp_version",    0),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        cv = st.session_state["comp_version"]

        # ── Formulário ───────────────────────────────────────────────
        st.markdown("#### Local de utilização")
        local_comp = st.selectbox("", [
            "Construção Civil", "Agronegócio", "Mineração",
            "Indústria", "Escritório Administrativo",
            "Call Center / Teleatendimento",
            "Infraestrutura / Obras Externas",
            "Trabalho a Céu Aberto (NR-21)",
            "Outro",
        ], key=f"local_comp_v{cv}", label_visibility="collapsed")

        st.divider()

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("##### Produto A")
            nome_a  = st.text_input(
                "Nome / modelo", key=f"comp_nome_a_v{cv}",
                placeholder="Ex: Beliche metálico padrão",
            )
            specs_a = st.text_area(
                "Especificações técnicas", key=f"comp_specs_a_v{cv}",
                height=120,
                placeholder="Material, dimensões, características...",
            )
        with col_p2:
            st.markdown("##### Produto B")
            nome_b  = st.text_input(
                "Nome / modelo", key=f"comp_nome_b_v{cv}",
                placeholder="Ex: Beliche metálico reforçado NR",
            )
            specs_b = st.text_area(
                "Especificações técnicas", key=f"comp_specs_b_v{cv}",
                height=120,
                placeholder="Material, dimensões, características...",
            )

        st.divider()

        col_cbtn1, col_cbtn2 = st.columns([3, 1])
        with col_cbtn1:
            comparar = st.button("⚖️ Comparar produtos", use_container_width=True)
        with col_cbtn2:
            if st.button("🗑️ Limpar", use_container_width=True, key="limpar_comp"):
                st.session_state["resultado_comp"] = ""
                st.session_state["comp_version"]  += 1
                st.rerun()

        # ── Comparação ───────────────────────────────────────────────
        if comparar:
            if not nome_a.strip() or not nome_b.strip():
                st.warning("Preencha o nome dos dois produtos para comparar.")
            else:
                with st.container():
                    st.session_state["resultado_comp"] = _executar_com_progresso(
                        engine.comparar_produtos,
                        [
                            "Preparando comparação...",
                            "Buscando NRs para ambos os produtos...",
                            "Priorizando NR-24, NR-18, NR-31...",
                            "Analisando Produto A...",
                            "Analisando Produto B...",
                            "Gerando recomendação final...",
                        ],
                        produto_a={"nome": nome_a.strip(), "specs": specs_a.strip()},
                        produto_b={"nome": nome_b.strip(), "specs": specs_b.strip()},
                        local=local_comp,
                    )

        # ── Resultado ────────────────────────────────────────────────
        if st.session_state["resultado_comp"]:
            st.divider()
            st.markdown("### ⚖️ Resultado da comparação")
            st.markdown(st.session_state["resultado_comp"])

    # ════════════════════════════════════════════════════════════════
    # ABA 3 — CONSULTORIA
    # ════════════════════════════════════════════════════════════════
    with aba_consultoria:

        # ── Inicializa estado ────────────────────────────────────────
        if "chat_historico" not in st.session_state:
            st.session_state["chat_historico"] = []
        if "chat_version" not in st.session_state:
            st.session_state["chat_version"] = 0

        cv = st.session_state["chat_version"]

        st.markdown("#### 💬 Consultoria de NRs")
        st.caption("Tire dúvidas sobre legislação, dimensionamento de espaços e o que é obrigatório em alojamentos, vestiários e refeitórios.")

        col_c1, col_c2 = st.columns([3, 1])
        with col_c2:
            if st.button("🗑️ Limpar conversa", use_container_width=True, key="limpar_chat"):
                st.session_state["chat_historico"] = []
                st.session_state["chat_version"] += 1
                st.rerun()

        st.divider()

        # ── Exibe histórico da conversa ──────────────────────────────
        for msg in st.session_state["chat_historico"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-role-user">👤 Vendedor</div>'
                    f'<div class="chat-msg-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-role-assistant">🤖 Consultor NR</div>'
                    f'<div class="chat-msg-assistant">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # ── Input da pergunta ────────────────────────────────────────
        st.divider()
        pergunta = st.text_area(
            "Sua pergunta",
            placeholder=(
                "Ex: Quantos colchões e beliches cabem num dormitório de 5m x 8m? "
                "O que mais é obrigatório nesse espaço pela NR-24?"
            ),
            height=90,
            key=f"pergunta_v{cv}",
        )

        if st.button("📨 Enviar pergunta", use_container_width=True, key=f"enviar_v{cv}"):
            if not pergunta.strip():
                st.warning("Digite uma pergunta antes de enviar.")
            else:
                try:
                    resposta = _executar_com_progresso(
                        engine.consultar,
                        [
                            "Buscando NRs relevantes...",
                            "Consultando NR-24, NR-18, NR-31...",
                            "Calculando e formulando resposta...",
                            "Finalizando...",
                        ],
                        pergunta=pergunta.strip(),
                        historico=st.session_state["chat_historico"],
                    )
                    st.session_state["chat_historico"].append(
                        {"role": "user", "content": pergunta.strip()}
                    )
                    st.session_state["chat_historico"].append(
                        {"role": "assistant", "content": resposta}
                    )
                    st.session_state["chat_version"] += 1
                    st.rerun()
                except Exception as e:
                    msg = str(e)
                    if "503" in msg or "UNAVAILABLE" in msg:
                        st.warning("⏳ Servidor sobrecarregado. Aguarde alguns segundos e tente novamente.")
                    else:
                        st.error(f"Erro inesperado: {msg}")

    # ════════════════════════════════════════════════════════════════
    # ABA 4 — HISTÓRICO
    # ════════════════════════════════════════════════════════════════
    with aba_historico:
        if not history:
            st.info("O histórico não está configurado. Configure GOOGLE_SHEETS_CREDS e GOOGLE_SHEET_ID no arquivo .env.")
        else:
            st.markdown("#### Buscar análises anteriores")
            busca = st.text_input(
                "Pesquisar por nome de produto",
                placeholder="Ex: beliche, armário, mesa..."
            )
            registros = history.buscar(busca)

            if not registros:
                if busca:
                    st.warning(f"Nenhuma análise encontrada para '{busca}'.")
                else:
                    st.caption("Digite um nome de produto para buscar no histórico.")
            else:
                st.caption(f"{len(registros)} registro(s) encontrado(s).")
                st.divider()
                for reg in registros:
                    with st.expander(f"📄 {reg['produtos_resumo']} — {reg['data']}"):
                        st.markdown(f"**Vendedor:** {reg['usuario']}")
                        st.markdown(f"**Local:** {reg['local']}")
                        st.markdown(f"**Produtos analisados:** {reg['produtos_resumo']}")
                        st.divider()
                        st.markdown(reg["resultado_texto"])


# ── Ponto de entrada ─────────────────────────────────────────────────────────
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if not st.session_state["autenticado"]:
    tela_login()
else:
    app_principal()