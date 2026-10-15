
# Sansad — Parliamentary Intelligence

### Parliamentary Participation Analysis and AI-Powered Research | 17th Lok Sabha (2019–2024)

Sansad is a parliamentary research project that combines exploratory data analysis, unsupervised machine learning, semantic retrieval, and a locally running language model to explore parliamentary participation and query official proceedings.

The project has two complementary components:

- **Parliamentary participation analysis:** explores attendance, questions, debates, Private Member Bills, and participation patterns among MPs.
- **Parliamentary AI Assistant:** retrieves relevant passages from the President's Address to Parliament and uses a local language model to generate source-grounded answers.

The goal is to explore how data analysis and retrieval-augmented generation (RAG) can support research into parliamentary activity.

---

## Features

### Parliamentary Participation Analysis

- Explore MP participation indicators.
- Examine attendance, questions, debates, and Private Member Bills.
- Compare participation patterns across parties, states, and minister status.
- Investigate relationships between participation indicators.
- Explore K-Means clustering and PCA visualizations.

### Parliamentary AI Assistant

- Ask natural-language questions about parliamentary proceedings.
- Retrieve relevant passages using semantic similarity.
- Generate answers using a locally running Ollama language model.
- Display retrieved PDF page references and source passages for inspection.
- Explore the underlying parliamentary text and retrieval pipeline.

The assistant is a research prototype. Retrieved passages can be incomplete or irrelevant, and generated answers may contain errors. Source passages should be checked before relying on an answer.

---

## Project 1: Parliamentary Participation Analysis

### Objectives

1. Examine the demographic composition of the 17th Lok Sabha.
2. Analyze attendance and parliamentary participation distributions.
3. Compare participation patterns across parties and states.
4. Examine relationships between participation indicators.
5. Identify exploratory participation profiles using K-Means clustering.
6. Visualize clustering structure using Principal Component Analysis (PCA).
7. Evaluate clustering solutions using silhouette analysis.

### Dataset

The structured dataset is the `india-representatives-activity` dataset by Vonter, which compiles parliamentary activity data sourced from PRS India.

The original dataset is stored locally as:

`data/raw/17th.csv`

The analysis uses indicators including attendance, debates, questions, and Private Member Bills.

### Data Sources

- Vonter — India Representatives Activity dataset
- PRS India — MP Track / 17th Lok Sabha
- Digital Sansad — Parliament of India

The structured dataset supports the initial analysis. PRS India and Digital Sansad provide parliamentary context and reference material.

### Methodology

#### Data Preparation

`01_Data_Preparation.ipynb` covers:

- Dataset loading and structural inspection
- Missing-value analysis and duplicate checks
- Data-type conversion and date processing
- Preparation of attendance and participation variables
- Final data-quality checks

#### Exploratory Data Analysis

`02_EDA.ipynb` examines:

- Age and gender distributions
- Attendance, questions, debates, and Private Member Bills
- Participation differences across ministers, parties, and states
- Relationships and correlations between participation indicators

#### Unsupervised Learning

K-Means clustering was applied to attendance, questions, debates, and Private Member Bills. Features were standardized before clustering.

PCA was used to visualize cluster structure in two dimensions, and silhouette analysis compared candidate cluster counts from 2 to 6.

### Selected Findings

#### Participation distributions

Questions, debates, and Private Member Bills showed strongly right-skewed distributions, indicating that activity levels varied substantially between representatives.

#### Relationships

Attendance had positive but relatively weak linear associations with:

| Indicator | Approximate Pearson correlation |
|---|---:|
| Questions | 0.15 |
| Debates | 0.22 |
| Private Member Bills | 0.23 |

These associations suggest that attendance alone does not strongly explain variation in the other indicators. Correlation does not establish causation.

#### K-Means clustering

An exploratory four-cluster solution identified different participation profiles, including a large group with relatively high attendance and moderate participation, a smaller group with higher activity counts, a group with comparatively lower activity, and a very small group with exceptionally high debate participation.

Silhouette analysis produced the following results:

| Number of clusters | Silhouette score |
|---|---:|
| 2 | 0.517 |
| 3 | 0.339 |
| 4 | 0.350 |
| 5 | 0.337 |
| 6 | 0.304 |

Among the tested configurations, two clusters achieved the highest silhouette score. The four-cluster solution is therefore treated as exploratory segmentation, not an objective ranking of MPs.

---

## Project 2: Parliamentary AI Assistant

### Approach

The assistant implements a local retrieval-augmented generation workflow:

1. Extract and clean parliamentary text.
2. Split the text into overlapping chunks.
3. Convert chunks into vector embeddings.
4. Store the vectors in a FAISS index.
5. Retrieve relevant passages for a user question.
6. Pass the retrieved context to a local language model.
7. Display the answer alongside the retrieved source pages.

### Current implementation

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Data processing | Python |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Local language model | Ollama with `llama3.2:3b` |
| RAG framework | LangChain |

The current corpus is the English portion of the President's Address in the parliamentary proceedings PDF `Floor.pdf` for 31 January 2024.

The Hindi extraction was not used in the current English-language index because its extracted text was noisy.

### AI notebooks

- `01_Parliamentary_AI_Experiments.ipynb` — experiments and development work.
- `02_Parliamentary_RAG_Pipeline.ipynb` — the structured retrieval and generation pipeline.

These notebooks are currently maintained in the separate local `Lok_Sabha_AI` project folder. Their final location and paths will be updated when the project folders are consolidated.

### Limitations

- The current RAG corpus covers a limited selection of proceedings, not every parliamentary debate or document.
- Retrieval quality depends on the question, text extraction, chunking, and embedding similarity.
- Retrieved passages may not fully support a generated answer.
- The model runs locally and may produce inaccurate or incomplete responses.
- MP participation counts should be interpreted in the context of the dataset's coverage and definitions.

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Ollama

---

## Running the AI Assistant

The application requires Python dependencies, the cleaned MP dataset, the processed parliamentary text, the FAISS index, and a running Ollama installation with the required model.

### 1. Install dependencies

Install the packages listed in `requirements.txt` after the file has been finalized for the combined project.

### 2. Install and start Ollama

Install Ollama from its official website:

https://ollama.com/

Pull the model used by the current application:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before launching the app.

### 3. Start Streamlit

From the project root, run:

```bash
streamlit run app.py
```

**Note:** The repository setup instructions and file paths are being finalized as the two local project folders are consolidated. The application may require local data preparation and FAISS index generation before it can run on a fresh installation.

---

## Project Structure

The target structure after consolidation is:

```text
17th-lok-sabha-analysis/
├── app.py
├── styles.css
├── assets/
│   └── parliament.jpg
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 01_Parliamentary_AI_Experiments.ipynb
│   └── 02_Parliamentary_RAG_Pipeline.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

This represents the planned consolidated structure; local data and generated indexes may be excluded from version control and rebuilt when needed.

---

## Future Work

- Improve retrieval relevance and answer grounding.
- Evaluate retrieval quality on a curated set of parliamentary questions.
- Expand the document corpus to additional proceedings.
- Add richer source metadata and clearer citations.
- Improve the user interface and interaction design.
- Explore additional NLP techniques for parliamentary questions and debates.

---

## Disclaimer

This project is an independent educational and research prototype. It is not an official Parliament of India application. Parliamentary data and generated answers should be checked against the cited source material before being used for formal research or reporting.