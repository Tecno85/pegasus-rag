"""Streamlit entry point for the public Pegasus RAG demonstration."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import streamlit as st

from pegasus_rag.chunking import chunk_sections
from pegasus_rag.config import Settings
from pegasus_rag.corpus import build_base_index, load_or_build_base_index
from pegasus_rag.embeddings import LocalSentenceTransformer
from pegasus_rag.errors import PegasusError
from pegasus_rag.generator import GeminiGenerator
from pegasus_rag.loaders import load_document
from pegasus_rag.service import RagService
from pegasus_rag.store import VectorIndex

ROOT_DIR = Path(__file__).resolve().parent
WING_MARK_PATH = ROOT_DIR / "assets" / "pegasus-wing-mark.png"
WING_HERO_PATH = ROOT_DIR / "assets" / "pegasus-wing-hero.png"


def image_data_uri(path: Path) -> str:
    """Return a local PNG as a browser-safe data URI."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


wing_mark_uri = image_data_uri(WING_MARK_PATH)
wing_hero_uri = image_data_uri(WING_HERO_PATH)

st.set_page_config(
    page_title="Pegasus RAG",
    page_icon=str(WING_MARK_PATH),
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

      :root {{
        --ink-950: #03101f;
        --ink-900: #061324;
        --ink-850: #09182b;
        --ink-800: #0c1d33;
        --line: rgba(153, 174, 207, .2);
        --line-strong: rgba(153, 174, 207, .34);
        --paper: #f5f2eb;
        --muted: #b8c4d6;
        --blue: #2f76ff;
        --blue-bright: #4b8aff;
        --coral: #ff675e;
        --success: #38d98a;
      }}

      html, body, [class*="css"] {{ font-family: "Manrope", sans-serif; }}
      .stApp {{
        color: var(--paper);
        background: var(--ink-900);
      }}
      [data-testid="stHeader"] {{ background: rgba(3, 16, 31, .88); }}
      [data-testid="stDecoration"] {{ display: none; }}
      [data-testid="stToolbar"] {{ color: var(--muted); }}
      [data-testid="stMainBlockContainer"] {{
        max-width: 1240px;
        padding-top: 2.25rem;
        padding-bottom: 5rem;
      }}

      [data-testid="stSidebar"] {{
        background: var(--ink-950);
        border-right: 1px solid var(--line);
      }}
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding: 1.45rem 1.35rem 2rem;
      }}
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] small {{
        color: var(--muted);
        font-size: .91rem;
        line-height: 1.55;
      }}
      [data-testid="stSidebar"] hr {{ border-color: var(--line); margin: 1.35rem 0; }}

      .brand {{ display: flex; align-items: center; gap: .8rem; margin: .15rem 0 2rem; }}
      .brand img {{ width: 38px; height: 38px; object-fit: cover; border-radius: 7px; }}
      .brand strong {{
        display: block; color: var(--paper); font-size: 1.2rem; letter-spacing: -.025em;
      }}
      .brand span {{ color: #91a2ba; font-size: .86rem; }}
      .section-label {{
        color: #a9b7ca; font-size: .875rem; font-weight: 700; letter-spacing: .09em;
        text-transform: uppercase; margin: .2rem 0 .65rem;
      }}
      .status-block {{
        padding: .25rem 0 1.25rem; border-bottom: 1px solid var(--line); margin-bottom: 1.4rem;
      }}
      .status-title {{
        display: flex; gap: .7rem; align-items: center; font-size: 1.05rem; color: var(--paper);
      }}
      .status-dot {{ width: 11px; height: 11px; border-radius: 50%; background: var(--blue);
                     box-shadow: 0 0 0 4px rgba(47,118,255,.12); }}
      .status-meta {{ color: var(--muted); font-size: .92rem; margin: .4rem 0 0 1.7rem; }}
      .model-block {{ padding: .2rem 0; }}
      .model-name {{ color: var(--paper); font-size: 1rem; font-weight: 700; margin-top: .25rem; }}
      .model-name::after {{ content: ""; display: inline-block; width: 8px; height: 8px;
                           border-radius: 50%; background: var(--success); margin-left: .55rem; }}
      .model-id {{ color: #a9b7ca; font-size: .9rem; margin-top: .2rem; overflow-wrap: anywhere; }}
      .privacy-note {{ border-left: 2px solid var(--blue); padding-left: .8rem; color: var(--muted);
                       font-size: .88rem; line-height: 1.6; }}

      [data-testid="stFileUploader"] {{ background: var(--ink-900); border-radius: .25rem; }}
      [data-testid="stFileUploaderDropzone"] {{
        background: transparent; border: 1px dashed var(--line-strong); border-radius: .35rem;
        min-height: 104px; padding: .85rem;
      }}
      [data-testid="stFileUploaderDropzone"] button {{
        background: transparent; border-color: var(--blue); color: var(--blue-bright);
        font-size: .9rem;
      }}
      [data-testid="stSidebar"] .stButton > button {{
        min-height: 2.75rem; border-radius: .3rem; border: 1px solid var(--line-strong);
        background: transparent; color: var(--paper); font-size: .9rem; font-weight: 600;
      }}
      [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: var(--blue); border-color: var(--blue); color: white;
      }}
      [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: var(--blue-bright); color: white; background: rgba(47,118,255,.1);
      }}

      .hero {{
        min-height: 345px;
        padding: 1rem 45% 1.5rem 0;
        background-image: url('{wing_hero_uri}');
        background-size: auto 100%;
        background-repeat: no-repeat;
        background-position: right top;
      }}
      .eyebrow {{
        color: var(--blue-bright); font-size: .875rem; font-weight: 800;
        letter-spacing: .1em; text-transform: uppercase;
      }}
      .hero h1 {{
        color: var(--paper); margin: 1.15rem 0 1rem; max-width: 700px;
        font-size: clamp(2.8rem, 5vw, 4.5rem); font-weight: 700;
        line-height: 1.02; letter-spacing: -.055em;
      }}
      .hero h1 .accent {{ color: var(--blue-bright); }}
      .hero p {{ color: var(--muted); max-width: 40rem; font-size: 1.05rem; line-height: 1.65; }}

      .composer-label {{
        margin: .25rem 0 .55rem; color: #a9b7ca; font-size: .875rem; font-weight: 800;
        letter-spacing: .1em; text-transform: uppercase;
      }}
      [data-testid="stForm"] {{ border: none; padding: 0; }}
      [data-testid="stForm"] [data-testid="stTextInputRootElement"] {{
        min-height: 4.25rem; border: 1px solid var(--blue); border-radius: .35rem;
        background: var(--ink-850); box-shadow: none;
      }}
      [data-testid="stForm"] input {{
        color: var(--paper); font-size: 1.02rem; padding-left: .75rem;
      }}
      [data-testid="stForm"] input::placeholder {{ color: #8798b0; opacity: 1; }}
      [data-testid="stForm"] .stButton > button {{
        min-height: 4.25rem; border-radius: .35rem; border-color: var(--blue);
        background: var(--blue); color: white; font-size: .96rem; font-weight: 800;
      }}
      [data-testid="stForm"] .stButton > button:hover {{
        background: var(--blue-bright); border-color: var(--blue-bright);
      }}
      [data-testid="stFormSubmitButton"] button {{
        min-height: 4.25rem; border-radius: .35rem; border-color: var(--blue);
        background: var(--blue); color: white; font-size: .96rem; font-weight: 800;
      }}
      [data-testid="stFormSubmitButton"] button:hover {{
        background: var(--blue-bright); border-color: var(--blue-bright); color: white;
      }}
      .composer-tip {{
        color: #a4b3c7; text-align: right; font-size: .9rem; margin: -.25rem 0 2rem;
      }}

      .content-label {{
        margin: .5rem 0 .8rem; color: var(--blue-bright); font-size: .875rem;
        font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
      }}
      .citation-label {{ color: var(--coral); }}
      .stMain .stButton > button[kind="secondary"] {{
        min-height: 3.9rem; justify-content: flex-start; text-align: left; white-space: normal;
        background: transparent; color: var(--paper); border: 0;
        border-bottom: 1px solid var(--line);
        border-radius: 0; font-size: .96rem; line-height: 1.4; padding: .75rem .35rem;
      }}
      .stMain .stButton > button[kind="secondary"]:hover {{
        color: white; border-bottom-color: var(--blue); background: rgba(47,118,255,.06);
      }}
      .evidence-preview {{
        border-left: 2px solid var(--coral); padding: .15rem 0 .3rem 1.25rem; min-height: 210px;
      }}
      .evidence-preview strong {{ color: var(--paper); font-size: 1rem; }}
      .evidence-preview p {{
        color: var(--muted); font-size: .93rem; line-height: 1.55; margin: .6rem 0;
      }}
      .evidence-preview .evidence-meta {{ color: #afbdd0; font-size: .9rem; }}
      .evidence-preview .coral {{ color: var(--coral); font-weight: 700; }}

      [data-testid="stChatMessage"] {{
        border: 1px solid var(--line); background: var(--ink-850); border-radius: .55rem;
        padding: .45rem .75rem; margin-bottom: .8rem;
      }}
      [data-testid="stChatMessage"] p {{ color: #d8dfeb; font-size: 1rem; line-height: 1.65; }}
      [data-testid="stExpander"] {{
        background: var(--ink-800); border-color: var(--line); border-radius: .35rem;
      }}
      .source-card {{
        border-left: 2px solid var(--coral); padding: .75rem 1rem; margin: .5rem 0 1rem;
        background: rgba(255,103,94,.035); color: #d5ddea; font-size: .94rem; line-height: 1.6;
      }}
      a {{ color: var(--blue-bright) !important; }}
      footer {{ visibility: hidden; }}

      @media (max-width: 900px) {{
        [data-testid="stMainBlockContainer"] {{ padding: 1.2rem 1.1rem 4rem; }}
        .hero {{ min-height: auto; padding: .75rem 0 1rem; background-size: 58% auto;
                 background-position: 125% 0; }}
        .hero h1 {{ max-width: 78%; font-size: clamp(2.45rem, 11vw, 3.7rem); }}
        .hero p {{ max-width: 90%; font-size: 1rem; }}
      }}
      @media (max-width: 600px) {{
        .hero {{ background-image: none; }}
        .hero {{ padding-top: 2.4rem; }}
        .hero h1 {{ max-width: 100%; font-size: 2.65rem; }}
        .composer-tip {{ text-align: left; margin-top: .25rem; }}
        .evidence-preview {{ min-height: auto; margin-bottom: 1rem; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> LocalSentenceTransformer:
    return LocalSentenceTransformer(model_name)


@st.cache_resource(show_spinner="Preparando la base documental (la primera carga puede tardar)…")
def get_base_index(model_name: str) -> VectorIndex:
    settings = get_settings()
    return load_or_build_base_index(settings, get_embedder(model_name))


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("upload_index", None)
    st.session_state.setdefault("processed_names", [])


def source_panel(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Fuentes consultadas ({len(sources)})"):
        for source in sources:
            label = f"Fuente {source['number']} · {source['source']} · {source['location']}"
            if source.get("source_url"):
                st.markdown(f"**[{label}]({source['source_url']})**")
            else:
                st.markdown(f"**{label}**")
            st.caption(f"Similitud: {source['score']:.1%}")
            st.markdown(
                f"<div class='source-card'>{escape(source['excerpt'])}</div>",
                unsafe_allow_html=True,
            )


def process_uploads(uploaded_files, settings: Settings) -> None:
    if not uploaded_files:
        raise ValueError("Selecciona al menos un documento.")
    if len(uploaded_files) > settings.max_upload_files:
        raise ValueError(f"Puedes procesar máximo {settings.max_upload_files} archivos.")
    sections = []
    for uploaded in uploaded_files:
        sections.extend(
            load_document(
                uploaded.name,
                uploaded.getvalue(),
                max_size_mb=settings.max_upload_mb,
            )
        )
    chunks = chunk_sections(
        sections,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    st.session_state.upload_index = VectorIndex.build(
        chunks, get_embedder(settings.embedding_model)
    )
    st.session_state.processed_names = [uploaded.name for uploaded in uploaded_files]


initialize_session()
settings = get_settings()
base_index_error = None
try:
    base_index = get_base_index(settings.embedding_model)
except Exception as exc:
    base_index = None
    base_index_error = str(exc)

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
          <img src="{wing_mark_uri}" alt="Marca de ala plegada de Pegasus RAG">
          <div><strong>Pegasus RAG</strong><span>Ala de Papel</span></div>
        </div>
        <div class="section-label">Base de conocimiento</div>
        """,
        unsafe_allow_html=True,
    )
    if base_index:
        st.markdown(
            f"""
            <div class="status-block">
              <div class="status-title">
                <span class="status-dot"></span><strong>Base lista</strong>
              </div>
              <div class="status-meta">{len(base_index.chunks)} fragmentos indexados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("La base inicial todavía no está indexada.")
        if base_index_error:
            st.caption(f"Detalle: {base_index_error}")

    st.markdown('<div class="section-label">Añadir documentos</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Subir archivos",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True,
        help=(
            f"Máximo {settings.max_upload_files} archivos de {settings.max_upload_mb} MB. "
            "No se guardan al terminar la sesión."
        ),
    )
    if st.button("Procesar documentos", type="primary", use_container_width=True):
        try:
            with st.spinner("Leyendo y generando embeddings locales…"):
                process_uploads(uploaded_files, settings)
            st.success("Documentos listos para consultar.")
        except (PegasusError, ValueError) as exc:
            st.error(str(exc))

    if st.session_state.processed_names:
        st.caption("Temporales: " + ", ".join(st.session_state.processed_names))
        if st.button("Eliminar documentos temporales", use_container_width=True):
            st.session_state.upload_index = None
            st.session_state.processed_names = []
            st.rerun()

    st.divider()
    st.markdown('<div class="section-label">Herramientas</div>', unsafe_allow_html=True)
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reconstruir base documental", use_container_width=True):
        try:
            with st.spinner("Descargando, verificando e indexando los cinco manuales…"):
                build_base_index(
                    settings,
                    get_embedder(settings.embedding_model),
                    force_download=True,
                )
            get_base_index.clear()
            st.success("Base reconstruida.")
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudo reconstruir la base: {exc}")

    st.divider()
    st.markdown('<div class="section-label">Modelo configurado</div>', unsafe_allow_html=True)
    if settings.gemini_api_key:
        st.markdown(
            f"<div class='model-block'><div class='model-name'>Gemini</div>"
            f"<div class='model-id'>{escape(settings.gemini_model)}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Configura GEMINI_API_KEY para generar respuestas.")
    st.markdown(
        "<div class='privacy-note'>Las cargas viven solo en memoria. "
        "No subas información sensible al demo público.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Conocimiento interno, sin abrir manuales</div>
      <h1>Pregunta. Pegasus encuentra la <span class="accent">evidencia.</span></h1>
      <p>Consulta onboarding, ingeniería, arquitectura e incidentes. Cada respuesta está
      respaldada por fragmentos verificables de la documentación.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

suggestions = [
    "¿Cuál es la cobertura mínima exigida para un Code Review?",
    "¿Cuántas aprobaciones necesita un Pull Request?",
    "¿Qué tiempo de respuesta tiene un incidente SEV-1?",
    "¿Cómo se propaga el Trace ID entre microservicios?",
]
st.markdown('<div class="composer-label">Tu pregunta</div>', unsafe_allow_html=True)
with st.form("question-form", clear_on_submit=True, border=False):
    input_column, action_column = st.columns([8.2, 1.8], vertical_alignment="bottom")
    with input_column:
        typed_question = st.text_input(
            "Pregunta sobre los documentos",
            placeholder="Escribe tu pregunta con precisión…",
            label_visibility="collapsed",
        )
    with action_column:
        submitted = st.form_submit_button(
            "Preguntar",
            icon=":material/arrow_forward:",
            use_container_width=True,
        )
st.markdown(
    '<div class="composer-tip">Consejo: sé específico sobre el tema, servicio o incidente.</div>',
    unsafe_allow_html=True,
)

suggested_question = None
if not st.session_state.messages:
    suggestions_column, evidence_column = st.columns([1.08, 1], gap="large")
    with suggestions_column:
        st.markdown('<div class="content-label">Prueba una pregunta</div>', unsafe_allow_html=True)
        for index, suggestion in enumerate(suggestions, start=1):
            number_column, question_column = st.columns([0.13, 0.87], vertical_alignment="center")
            with number_column:
                st.markdown(
                    f"<div style='color:var(--blue-bright);font-size:1.65rem;"
                    f"font-weight:600'>{index:02d}</div>",
                    unsafe_allow_html=True,
                )
            with question_column:
                if st.button(
                    suggestion,
                    key=f"suggestion-{index}",
                    use_container_width=True,
                ):
                    suggested_question = suggestion
    with evidence_column:
        st.markdown(
            """
            <div class="content-label citation-label">Así citamos</div>
            <div class="evidence-preview">
              <strong>Fuente verificable</strong>
              <p>Cada respuesta identifica el documento y la ubicación exacta utilizada.</p>
              <div class="evidence-meta">Documento · Página o sección · Similitud</div>
              <p>“Aquí podrás desplegar el fragmento que respalda la respuesta.”</p>
              <span class="coral">Evidencia visible al responder</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            source_panel(message.get("sources", []))

question = (typed_question if submitted else None) or suggested_question
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    indexes = [index for index in (base_index, st.session_state.upload_index) if index]
    if not indexes:
        error_text = "Primero construye la base documental o procesa un archivo."
        st.error(error_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_text, "sources": []}
        )
    else:
        generator = GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
        service = RagService(
            generator,
            top_k=settings.top_k,
            threshold=settings.similarity_threshold,
        )
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[:-1]
        ]
        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando evidencia y redactando la respuesta…"):
                    answer = service.ask(question, indexes, history)
                st.markdown(answer.text)
                serialized_sources = [source.to_dict() for source in answer.sources]
                source_panel(serialized_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.text,
                        "sources": serialized_sources,
                    }
                )
            except PegasusError as exc:
                message = f"No pude completar la respuesta: {exc}"
                st.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message, "sources": []}
                )
