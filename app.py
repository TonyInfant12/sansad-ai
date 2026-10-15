from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Page configuration
st.set_page_config(
    page_title="Sansad | Parliamentary Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Project paths
APP_DIR = Path(__file__).parent
MP_DATA_PATH = (
    APP_DIR
    / "data"
    / "processed"
    / "17th_cleaned.csv"
)

RAG_DATA_PATH = (
    APP_DIR
    / "data"
    / "processed"
    / "president_address_en_cleaned.jsonl"
)
FAISS_INDEX_PATH = APP_DIR / "data" / "processed" / "faiss_president_address"
CSS_PATH = APP_DIR / "styles.css"
HERO_IMAGE_PATH = APP_DIR / "assets" / "parliament.jpg"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:3b"


# Load the external stylesheet
if CSS_PATH.exists():
    st.html(CSS_PATH)
else:
    st.warning(
        "styles.css has not been created yet. "
        "The application will run with Streamlit's default styling."
    )


# Load the MP dataset
@st.cache_data
def load_data():
    df = pd.read_csv(MP_DATA_PATH)

    if "Attendance" in df.columns:
        df["Attendance_pct"] = pd.to_numeric(
            df["Attendance"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )

    return df


# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


# Load the FAISS index
@st.cache_resource
def load_vector_store():
    embeddings = load_embedding_model()

    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# Load the language model
@st.cache_resource
def load_llm():
    return ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0,
    )


# Prepare the parliamentary research prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a parliamentary research assistant.

Answer using only the supplied source passages.
Do not invent facts or add unsupported claims.
If the evidence is insufficient, say so.
Do not write page citations in the answer.
The application will list the retrieved source pages separately.

SOURCE PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
""")


# Retrieve source passages and generate an answer
def retrieve_and_answer(question, k=3):
    vector_store = load_vector_store()
    llm = load_llm()

    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join(
        f"[PDF page {doc.metadata['pdf_page']}]\n{doc.page_content}"
        for doc in docs
    )

    response = llm.invoke(
        rag_prompt.format_messages(
            context=context,
            question=question,
        )
    )

    answer = response.content.strip()

    source_pages = sorted({
        int(doc.metadata["pdf_page"])
        for doc in docs
        if doc.metadata.get("pdf_page") is not None
    })

    return answer, docs, source_pages


# Display the application header
st.markdown(
    """
    <section class="hero">
        <div class="hero-eyebrow">
            PARLIAMENTARY INTELLIGENCE
        </div>
        <h1>Sansad<span>.</span></h1>
        <p class="hero-subtitle">
            Explore parliamentary activity. Discover the record.
            Ask questions grounded in official proceedings.
        </p>
        <div class="hero-meta">
            <span>17th Lok Sabha</span>
            <span class="hero-meta-dot">•</span>
            <span>2019–2024</span>
            <span class="hero-meta-dot">•</span>
            <span>Research workspace</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Display the hero image when available
if HERO_IMAGE_PATH.exists():
    st.image(
        str(HERO_IMAGE_PATH),
        use_container_width=True,
    )

st.markdown(
    """
    <div class="section-intro">
        <div class="section-kicker">THE WORKSPACE</div>
        <h2>Parliament, made explorable.</h2>
        <p>
            Search member records, examine participation, and query
            parliamentary proceedings through a local AI research assistant.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigate between the application sections
mp_tab, qa_tab, about_tab = st.tabs([
    "01  Explore MPs",
    "02  Parliamentary Q&A",
    "03  About the project",
])


# Explore MPs
with mp_tab:
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-kicker">MEMBER DIRECTORY</div>
            <h2>Explore the representatives.</h2>
            <p>Filter the dataset to investigate participation and activity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            f"Could not find the MP dataset at:\n\n{MP_DATA_PATH}"
        )
        st.stop()

    total_records = len(df)
    total_parties = (
        df["Party"].nunique()
        if "Party" in df.columns
        else 0
    )
    total_states = (
        df["State"].nunique()
        if "State" in df.columns
        else 0
    )

    # Summary metrics
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("MP records", f"{total_records:,}")

    with metric2:
        st.metric("Political parties", f"{total_parties:,}")

    with metric3:
        st.metric("States and UTs", f"{total_states:,}")

    st.markdown("#### Refine your search")

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        search_name = st.text_input(
            "Search by name",
            placeholder="Enter an MP's name...",
        )

    with filter2:
        state_options = (
            sorted(df["State"].dropna().astype(str).unique())
            if "State" in df.columns
            else []
        )
        selected_state = st.selectbox(
            "State or Union Territory",
            ["All"] + state_options,
        )

    with filter3:
        party_options = (
            sorted(df["Party"].dropna().astype(str).unique())
            if "Party" in df.columns
            else []
        )
        selected_party = st.selectbox(
            "Political party",
            ["All"] + party_options,
        )

    minister_options = ["All", "Minister", "Non-minister"]
    selected_minister = st.selectbox(
        "Minister status",
        minister_options,
    )

    filtered_df = df.copy()

    if search_name and "Name" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Name"].astype(str).str.contains(
                search_name,
                case=False,
                na=False,
            )
        ]

    if selected_state != "All" and "State" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["State"].astype(str) == selected_state
        ]

    if selected_party != "All" and "Party" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Party"].astype(str) == selected_party
        ]

    if selected_minister != "All" and "Minister" in filtered_df.columns:
        minister_values = filtered_df["Minister"].astype(str).str.strip().str.lower()

        if selected_minister == "Minister":
            filtered_df = filtered_df[
                minister_values.isin(["yes", "true", "1", "minister"])
            ]
        else:
            filtered_df = filtered_df[
                ~minister_values.isin(["yes", "true", "1", "minister"])
            ]

    st.caption(f"{len(filtered_df):,} matching records")

    preferred_columns = [
        "Name",
        "Constituency",
        "Party",
        "State",
        "Minister",
        "Attendance",
        "Debates",
        "Questions",
        "Private Member Bills",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Participation overview
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-kicker">PARTICIPATION OVERVIEW</div>
            <h2>What does the dataset show?</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        if "Attendance_pct" in filtered_df.columns:
            average_attendance = filtered_df["Attendance_pct"].mean()
            value = (
                f"{average_attendance:.1f}%"
                if pd.notna(average_attendance)
                else "N/A"
            )
            st.metric("Average attendance", value)
        else:
            st.metric("Average attendance", "N/A")

    with stat2:
        if "Questions" in filtered_df.columns:
            questions = pd.to_numeric(
                filtered_df["Questions"],
                errors="coerce",
            )
            average_questions = questions.mean()
            value = (
                f"{average_questions:.1f}"
                if pd.notna(average_questions)
                else "N/A"
            )
            st.metric("Average questions", value)
        else:
            st.metric("Average questions", "N/A")

    with stat3:
        if "Debates" in filtered_df.columns:
            debates = pd.to_numeric(
                filtered_df["Debates"],
                errors="coerce",
            )
            average_debates = debates.mean()
            value = (
                f"{average_debates:.1f}"
                if pd.notna(average_debates)
                else "N/A"
            )
            st.metric("Average debates", value)
        else:
            st.metric("Average debates", "N/A")

    st.caption(
        "These are descriptive statistics for the filtered records. "
        "They do not, by themselves, measure an MP's overall effectiveness."
    )


# Parliamentary Q&A
with qa_tab:
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-kicker">RESEARCH ASSISTANT</div>
            <h2>Ask the parliamentary record.</h2>
            <p>
                Ask about the President's Address delivered on
                31 January 2024. Answers are generated from retrieved
                source passages in the indexed document.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice-panel">
            <strong>Source collection</strong>
            <p>
                President's Address to the joint sitting of Parliament,
                English text extracted from the proceedings PDF.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What did the President say about India's "
            "space programme?"
        ),
        height=110,
        key="parliament_question",
    )

    col_button, col_note = st.columns([1, 3])

    with col_button:
        ask_button = st.button(
            "Generate answer",
            type="primary",
            use_container_width=True,
        )

    with col_note:
        st.caption(
            "The model runs locally through Ollama. Retrieved passages "
            "are shown below the answer for inspection."
        )

    if ask_button:
        if not question.strip():
            st.warning("Enter a question first.")

        elif not RAG_DATA_PATH.exists():
            st.error(
                f"Could not find the cleaned source file:\n\n{RAG_DATA_PATH}"
            )

        elif not (
            (FAISS_INDEX_PATH / "index.faiss").exists()
            and (FAISS_INDEX_PATH / "index.pkl").exists()
        ):
            st.error(
                "The FAISS index files were not found. "
                "Build the index in the RAG notebook first."
            )

        else:
            try:
                with st.spinner(
                    "Searching the proceedings and preparing an answer..."
                ):
                    answer, docs, source_pages = retrieve_and_answer(
                        question.strip(),
                        k=3,
                    )

                st.markdown("### Answer")
                st.markdown(
                    '<div class="answer-panel"></div>',
                    unsafe_allow_html=True,
                )
                st.write(answer)

                if source_pages:
                    page_labels = ", ".join(
                        f"PDF p. {page}" for page in source_pages
                    )
                    st.markdown("#### Retrieved source pages")
                    st.caption(page_labels)

                st.markdown("#### Inspect the evidence")

                for index, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("pdf_page", "Unknown")
                    with st.expander(
                        f"Passage {index} · PDF page {page}"
                    ):
                        st.write(doc.page_content)

                st.caption(
                    "Retrieved pages are evidence to inspect, not proof "
                    "that every sentence in the answer is supported. "
                    "Check the passages before relying on a claim."
                )

            except Exception as error:
                st.error(
                    "The question could not be processed. "
                    "Check that Ollama is running, the model is installed, "
                    "and the FAISS index is available."
                )
                st.exception(error)


# About the project
with about_tab:
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-kicker">ABOUT THIS PROJECT</div>
            <h2>Parliamentary records, made accessible.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This project combines structured data exploration with a
        retrieval-augmented generation (RAG) assistant.

        **Explore MPs** uses a cleaned dataset of the 17th Lok Sabha
        to investigate attendance and recorded parliamentary activity.

        **Parliamentary Q&A** retrieves passages from the English
        version of the President's Address and uses a local language
        model to draft answers grounded in those passages.
        """
    )

    st.markdown("#### Technology")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("**Data analysis**")
        st.write("Python · Pandas · Streamlit")

    with tech2:
        st.markdown("**Retrieval**")
        st.write("Sentence Transformers · FAISS")

    with tech3:
        st.markdown("**Language model**")
        st.write("LangChain · Ollama · Llama 3.2")

    st.markdown("#### Current limitations")

    st.write(
        "- The MP directory depends on the quality and completeness "
        "of the cleaned dataset."
    )
    st.write(
        "- Parliamentary Q&A currently covers the indexed English "
        "President's Address, not every Lok Sabha debate or proceeding."
    )
    st.write(
        "- Retrieval can return irrelevant passages, and the language "
        "model can still produce unsupported statements."
    )
    st.write(
        "- This is a prototype, not an official Parliament service."
    )

# Footer
st.markdown(
    """
    <div class="app-footer">
        <span>SANSAD · PARLIAMENTARY INTELLIGENCE</span>
        <span>Research prototype · 17th Lok Sabha</span>
    </div>
    """,
    unsafe_allow_html=True,
)
