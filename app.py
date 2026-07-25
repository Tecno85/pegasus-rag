"""Streamlit entry point for the public Pegasus RAG demonstration."""

from __future__ import annotations

import streamlit as st

from pegasus_rag.chunking import chunk_sections
from pegasus_rag.config import Settings
from pegasus_rag.corpus import build_base_index
from pegasus_rag.embeddings import LocalSentenceTransformer
from pegasus_rag.errors import PegasusError
from pegasus_rag.generator import GeminiGenerator
from pegasus_rag.loaders import load_document
from pegasus_rag.service import RagService
from pegasus_rag.store import VectorIndex, index_exists

st.set_page_config(
    page_title="Pegasus RAG",
    page_icon="🪽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --pegasus-cyan: #35d7d0; --pegasus-navy: #091525; --pegasus-blue: #142a46; }
      .stApp {
        background:
          radial-gradient(circle at 90% 5%, rgba(53,215,208,.12), transparent 28rem),
          linear-gradient(180deg, #07111f 0%, #0b1728 100%);
      }
      [data-testid="stSidebar"] { background: rgba(8, 22, 38, .96); }
      .hero { padding: .4rem 0 1.2rem; }
      .eyebrow { color: var(--pegasus-cyan); font-size: .76rem; font-weight: 700;
                 letter-spacing: .14em; text-transform: uppercase; }
      .hero h1 { margin: .25rem 0; font-size: clamp(2rem, 5vw, 3.5rem); letter-spacing: -.04em; }
      .hero p { color: #aebed1; max-width: 48rem; font-size: 1.05rem; }
      .status-card { border: 1px solid rgba(53,215,208,.22); background: rgba(18,42,70,.5);
                     padding: .8rem 1rem; border-radius: .8rem; margin-bottom: 1rem; }
      .source-card { border-left: 3px solid var(--pegasus-cyan); padding: .35rem .8rem;
                     background: rgba(20,42,70,.42); border-radius: 0 .5rem .5rem 0; }
      [data-testid="stChatMessage"] { border: 1px solid rgba(174,190,209,.12);
                                      background: rgba(11,28,48,.72); border-radius: 1rem; }
      .stButton > button { border-radius: .75rem; border-color: rgba(53,215,208,.35); }
      .stButton > button:hover { border-color: var(--pegasus-cyan); color: var(--pegasus-cyan); }
      footer { visibility: hidden; }
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


@st.cache_resource(show_spinner="Cargando la base documental…")
def get_base_index(model_name: str) -> VectorIndex | None:
    settings = get_settings()
    if not index_exists(settings.index_dir):
        return None
    return VectorIndex.load(
        settings.index_dir,
        get_embedder(model_name),
        expected_model=model_name,
    )


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
                f"<div class='source-card'>{source['excerpt']}</div>",
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
base_index = get_base_index(settings.embedding_model)

with st.sidebar:
    st.markdown("## 🪽 Pegasus RAG")
    st.caption("Base de conocimiento")
    if base_index:
        st.markdown(
            f"<div class='status-card'>🟢 <strong>Base lista</strong><br>"
            f"{len(base_index.chunks)} fragmentos indexados</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("La base inicial todavía no está indexada.")

    uploaded_files = st.file_uploader(
        "Añadir documentos a esta sesión",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True,
        help=(
            f"Máximo {settings.max_upload_files} archivos de {settings.max_upload_mb} MB. "
            "No se guardan al terminar la sesión."
        ),
    )
    if st.button("Procesar documentos", use_container_width=True):
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
    if settings.gemini_api_key:
        st.caption(f"Gemini configurado · `{settings.gemini_model}`")
    else:
        st.warning("Configura GEMINI_API_KEY para generar respuestas.")
    st.caption(
        "🔒 Las cargas viven solo en memoria. No subas información sensible al demo público."
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Conocimiento interno, sin abrir manuales</div>
      <h1>Pregunta. Pegasus encuentra la evidencia.</h1>
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
suggested_question = None
if not st.session_state.messages:
    st.caption("Prueba una pregunta")
    columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if columns[index % 2].button(
            suggestion,
            key=f"suggestion-{index}",
            use_container_width=True,
        ):
            suggested_question = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            source_panel(message.get("sources", []))

typed_question = st.chat_input("Pregunta sobre los documentos…")
question = typed_question or suggested_question
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
