
# Document Similarity Analysis and Chatbot 🚀

This project offers a **suite of Streamlit pages** enabling:

1. **Analysis** of similarity between documents or sentences (Main Page)
2. **Search** for the most similar sentence or file in a corpus (Document Search Page)
3. **Implementation** of a chatbot with two approaches (basic TF-IDF or advanced Sentence-BERT + QA CamemBERT) (Chatbot Page)

## ⭐ Table of Contents

* [Key Features](#key-features)
* [Project Structure](#project-structure)
* [Installation & Prerequisites](#installation--prerequisites)
* [Launch & Usage](#launch--usage)
* [Pages Details](#pages-details)
* [Possible Improvements](#possible-improvements)
* [Authors & Contributions](#authors--contributions)
* [License](#license)

## ✨ Key Features

* **Similarity Calculation** through:
  * **TF-IDF** + multiple distances (*Manhattan, Euclidean, Jaccard, Cosine, etc.*)
  * **Embeddings** (*Word2Vec, FastText*)
  * **Sentence-BERT** for semantic search
  * **QA Pipeline (CamemBERT)** for answer extraction

* **Document Comparison**: top *k* most similar files or sentences
* **Search** for the most similar sentence to a text query
* **Chatbot** with two modes:
  * *Basic* (**TF-IDF**, significant word, sentence return)
  * *Advanced* (**Sentence-BERT + QA** with **multi-sentence context**)
* **Visualizations**: **WordCloud**, heatmap, score distribution, etc.

## 📁 Project Structure

```bash
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── app_main.py        # Main Page: Similarity Analysis
├── app_search.py      # Document Search Page
├── app_chatbot.py     # Chatbot Page
├── data/              # (Optional) Example .txt files
└── ...
```

## ⚙️ Installation & Prerequisites

1. Clone this repository:
```bash
git clone https://github.com/YourName/Document-Similarity-Analysis-Tool.git
cd Document-Similarity-Analysis-Tool
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # (macOS/Linux)
# or venv\Scripts\activate # (Windows)
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages include:
* streamlit
* scikit-learn
* nltk
* chardet
* sentence_transformers
* transformers
* torch
* wordcloud
* matplotlib
* seaborn

4. Download NLTK resources (if needed):
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## 🚀 Launch & Usage

Each page can be launched individually:
```bash
streamlit run app_main.py         # Main Page
streamlit run app_search.py       # Document Search Page
streamlit run app_chatbot.py      # Chatbot Page
```

**Tip**: If you have a `streamlit_app.py` combining all three pages, simply run:
```bash
streamlit run streamlit_app.py
```
Then navigate between pages using the Streamlit sidebar.

## 🔍 Pages Details

### Main Page
**Purpose**: Acts as a laboratory for testing and comparing similarity methods.

**Features**:
* Text loading (manual/file)
* Preprocessing (stopwords, stemming)
* Choice between binary vs occurrence, normalization, multiple distances
* Display of distance/similarity matrices
* Search for closest sentence (top k)
* Visualizations: WordCloud, embeddings, heatmap, etc.

### Document Search Page (Workshop 1)
**Purpose**: Search within a folder of .txt files or a single file.

**Features**:
* Tokenization of all .txt files
* TF-IDF + cosine search (sentences or documents)
* File comparison
* Score distribution, bar chart, WordCloud (global or per file)

### Chatbot Page (Workshop 2)
**Purpose**: Answer questions based on uploaded text.

**Two approaches**:
1. **Basic**:
   * TF-IDF (question + sentences)
   * Cosine similarity, returns most similar sentence
   * Most significant word, politeness formulas, etc.

2. **Advanced**:
   * Sentence-BERT to find top k most relevant sentences
   * Concatenation of these k sentences to form extended context
   * Passage to QA pipeline (CamemBERT) for answer extraction

**Notable improvement**:
In the advanced approach, multiple sentences are now selected and concatenated to provide broader context to the QA model, improving answer relevance.

## 💡 Possible Improvements

* Advanced indexing (Faiss, etc.) for large document volumes
* Finer segmentation by paragraphs or sections
* More contextual chatbot (re-ranking, similarity thresholds, etc.)
* Cleaner UI for less technical usage

## 👥 Authors & Contributions

* **Tbahriti Mohammed**:

Feel free to suggest or submit pull requests!

## 📄 License

This project is distributed under the MIT License. See the LICENSE file for details.

---

Thank you for your interest in this project! 🤝

Feel free to open an issue for any questions or problems!
