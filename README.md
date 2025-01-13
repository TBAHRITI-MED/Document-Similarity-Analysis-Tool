Document Similarity Analysis and Chatbot 🚀
Bienvenue dans ce projet, qui propose un ensemble de pages Streamlit pour :

Analyser la similarité entre des documents ou des phrases.
Rechercher la phrase ou le fichier le plus similaire dans un corpus.
Proposer un chatbot à deux approches (basique ou avancée).
⭐ Sommaire
Fonctionnalités Principales
Arborescence du Projet
Installation et Prérequis
Lancement et Utilisation
Détails sur les Pages
Page Principale
Page Recherche de Documents (Atelier 1)
Page Chatbot (Atelier 2)
Améliorations Possibles
Auteurs et Contributions
Licence
✨ Fonctionnalités Principales
Calcul de similarité via :
TF-IDF + plusieurs distances (Manhattan, Euclidienne, Jaccard, Cosinus, etc.),
Embeddings (Word2Vec, FastText),
Sentence-BERT pour la recherche sémantique,
Pipeline QA (CamemBERT) pour extraire des réponses.
Comparaison de documents : top 
𝑘
k fichiers ou phrases les plus proches.
Recherche de la phrase la plus similaire à une requête textuelle.
Chatbot avec deux modes :
Basique (TF-IDF, mot significatif, retour d’une phrase),
Avancé (Sentence-BERT + QA avec contexte multi-phrases).
Visualisations : WordCloud, heatmap, distribution des scores, etc.
🗂 Arborescence du Projet
bash
Copier le code
.
├── README.md               # Ce fichier
├── requirements.txt        # Liste des dépendances Python
├── app_main.py             # Page Principale : Analyse de Similarité
├── app_search.py           # Page Recherche de Documents
├── app_chatbot.py          # Page Chatbot
├── data/                   # (Optionnel) Fichiers .txt d'exemple
└── ...
⚙️ Installation et Prérequis
Cloner ce dépôt :

bash
Copier le code
git clone https://github.com/VotreNom/Document-Similarity-Analysis-Tool.git
cd Document-Similarity-Analysis-Tool
Créer un environnement virtuel (recommandé) :

bash
Copier le code
python -m venv venv
source venv/bin/activate   # (macOS/Linux)
# ou venv\Scripts\activate # (Windows)
Installer les dépendances :

bash
Copier le code
pip install -r requirements.txt
Exemples de packages requis :
streamlit, scikit-learn, nltk, chardet, sentence_transformers, transformers, torch, wordcloud, matplotlib, seaborn, etc.

Télécharger les ressources NLTK (si nécessaire) :

python
Copier le code
import nltk
nltk.download("punkt")
nltk.download("stopwords")
# etc.
🚀 Lancement et Utilisation
Chaque page peut se lancer individuellement (selon ta structure de code) :

bash
Copier le code
streamlit run app_main.py         # Page Principale
streamlit run app_search.py       # Page Recherche de Documents
streamlit run app_chatbot.py      # Page Chatbot
Astuce : si tu as un fichier streamlit_app.py qui rassemble les trois pages, lance simplement :

bash
Copier le code
streamlit run streamlit_app.py
et navigue entre les pages via la barre latérale Streamlit.

🔎 Détails sur les Pages
Page Principale
But : Faire office de laboratoire pour tester et comparer les méthodes de similarité.
Fonctionnalités :
Chargement du texte (manuel/fichier),
Prétraitement (stopwords, stemming),
Choix entre binaire vs occurrence, normalisation, distances multiples,
Affichage des matrices de distance/similarité,
Recherche de la phrase la plus proche (top 
𝑘
k),
Visualisations : WordCloud, embeddings, heatmap, etc.
Page Recherche de Documents (Atelier 1)
But : Rechercher dans un dossier de .txt ou un fichier unique.
Fonctionnalités :
Tokenisation de tous les fichiers .txt,
Recherche TF-IDF + cosinus (phrases ou documents),
Comparaison d’un fichier vs d’autres,
Distribution des scores, bar chart, WordCloud (global ou fichier).
Page Chatbot (Atelier 2)
But : Répondre à une question en se basant sur le texte téléversé.
Deux approches :
Basique :
TF-IDF (question + phrases),
Similarité cosinus, renvoi de la phrase la plus similaire,
Mot le plus significatif, formules de politesse, etc.
Avancé :
Sentence-BERT pour trouver les 
top
 
k
top k phrases les plus pertinentes,
Concaténation de ces 
𝑘
k phrases pour former un \textit{contexte étendu},
Passage au pipeline QA (CamemBERT) pour extraction de la réponse.
💡 Améliorations Possibles
Indexation plus avancée (Faiss, etc.) pour de gros volumes de documents.
Segmentation plus fine par paragraphes ou sections.
Chatbot encore plus contextuel (re-ranking, seuils de similarité, etc.).
UI plus épurée pour un usage moins technique.
Auteurs et Contributions
Votre Nom : Conception générale, page principale, intégration TF-IDF/QA, etc.
Collaborateurs (exemples) :
Nom 1 : Correction de bugs, WordCloud, styles.
Nom 2 : Amélioration du pipeline QA et embeddings.
Pour toute suggestion ou pull request, n’hésitez pas !

Licence
Ce projet est distribué sous la MIT License (ou celle de votre choix). Consultez le fichier LICENSE si besoin.

Merci de votre intérêt pour ce projet 🤝
N’hésite pas à ouvrir une issue pour toute question ou problème !