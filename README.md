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

* **Document Search & Comparison**:
  * Search for similar sentences within a single file
  * Search across multiple files or an entire directory
  * Compare one file against a collection of files
  * Return top *k* most similar files to a chosen reference file
  * Flexible search scope (single file, multiple files, or complete directory)

* **Text Analysis & Processing**:
  * Sentence tokenization
  * Document preprocessing (stopwords, stemming)
  * Similarity scoring and ranking
  * Document-level and sentence-level comparisons

* **Interactive Search Features**:
  * User-defined *k* parameter for top similar results
  * Support for various file formats and encoding
  * Batch processing of multiple documents
  * Directory-wide search capabilities

* **Chatbot** with two modes:
  * *Basic* (**TF-IDF**, significant word, sentence return)
  * *Advanced* (**Sentence-BERT + QA** with **multi-sentence context**)

* **Visualizations**: 
  * **WordCloud**
  * Similarity heatmaps
  * Score distribution graphs
  * Document comparison matrices

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
**Purpose**: Acts as a laboratory for testing and comparing similarity methods between documents and sentences.

**Features**:
* **Text Input & Processing**:
  * Multiple input methods (manual text entry, file upload)
  * Advanced preprocessing (stopwords, stemming, lemmatization)
  * Support for multiple file formats and encodings
* **Similarity Analysis**:
  * Multiple similarity metrics (TF-IDF, embeddings)
  * Various distance measures (Cosine, Euclidean, Manhattan, Jaccard)
  * Binary vs frequency-based analysis options
  * Customizable normalization parameters
* **Visualization Tools**:
  * Interactive similarity/distance matrices
  * Dynamic WordCloud generation
  * Embedding visualizations
  * Custom heatmaps for document comparison
* **Search Functions**:
  * Top-k similar sentences search
  * Configurable similarity thresholds
  * Results ranking and sorting options

### Document Search Page (Workshop 1)
**Purpose**: Comprehensive document search and comparison platform.

**Features**:
* **File Management**:
  * Single file analysis
  * Multiple file comparison
  * Full directory processing
  * Recursive folder search
* **Search Capabilities**:
  * Cross-document sentence search
  * Similar document identification
  * Top-k most similar files ranking
  * Paragraph-level comparison
* **Analysis Tools**:
  * TF-IDF vectorization
  * Cosine similarity computation
  * Document similarity matrices
  * File-to-file comparison metrics
* **Visualization**:
  * Score distribution plots
  * Interactive bar charts
  * Global and per-file WordClouds
  * Similarity heatmaps

### Chatbot Page (Workshop 2)
**Purpose**: Intelligent question-answering system based on document content.

**Two Complementary Approaches**:
1. **Basic (TF-IDF Based)**:
   * Question preprocessing and vectorization
   * TF-IDF document representation
   * Cosine similarity matching
   * Intelligent sentence selection
   * Context-aware response formatting
   * Significant keyword identification
   * Natural language response generation
   * Politeness and conversation management

2. **Advanced (Neural-Based)**:
   * Sentence-BERT embeddings
   * Smart context window selection
   * Multi-sentence context building
   * Dynamic context length adjustment
   * CamemBERT-based answer extraction
   * Confidence score calculation
   * Answer relevance ranking
   * Response quality optimization

**Notable Improvements**:
* Enhanced context building through multi-sentence selection
* Improved answer relevance through broader context window
* Better handling of complex questions
* Seamless integration of both approaches
* Dynamic switching based on question complexity

## 💡 Possible Improvements
* Implementation of Faiss indexing for large-scale document processing
* Enhanced document segmentation (smart paragraph and section detection)
* Advanced chatbot features:
  * Context memory management
  * Answer re-ranking algorithms
  * Dynamic similarity thresholds
  * Question type classification
* User Interface enhancements:
  * Simplified workflow options
  * Batch processing interface
  * Result visualization improvements
  * Custom preprocessing options
* Performance optimizations for large document collections
* Multi-language support expansion

## 👥 Authors & Contributions

* **Tbahriti Mohammed**

Feel free to suggest or submit pull requests!

## 📄 License

This project is distributed under the MIT License. See the LICENSE file for details.

---

Thank you for your interest in this project! 🤝

Feel free to open an issue for any questions or problems!
