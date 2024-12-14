# Document Similarity Analyzer

Ce projet permet d'analyser, de comparer et de rechercher des similarités entre des documents texte. Il utilise diverses techniques de traitement du langage naturel (NLP), telles que **TF-IDF**, **Cosine Similarity**, et **Stemming**, pour fournir une analyse approfondie du contenu des documents. L'application est construite avec **Streamlit** pour une interface web interactive.

## Fonctionnalités

### 1. **Analyse de Documents**
- **Charger et tokeniser les fichiers** : Vous pouvez charger un dossier complet ou un fichier `.txt` et l'application le tokenisera en phrases.
- **Affichage des phrases** : Affiche les phrases extraites du texte, avec la possibilité de les masquer ou de les afficher via un bouton interactif.
- **Calcul de la similarité** : Utilise **TF-IDF** et **Cosine Similarity** pour trouver les phrases les plus similaires dans un ou plusieurs fichiers.
- **Nuage de mots** : Crée un nuage de mots pour une visualisation rapide des termes les plus fréquents dans les documents.

### 2. **Chatbot pour Similarité**
- **Génération de réponses similaires** : Un chatbot génère des réponses similaires basées sur ce que l'utilisateur écrit. Il fonctionne en utilisant la **similarité cosinus** avec des phrases extraites des documents.
- **Exploration des mots similaires** : Permet à l'utilisateur de voir les mots les plus similaires à un mot donné en utilisant **Word2Vec** ou **FastText**.

### 3. **Prétraitement du Texte**
- **Suppression des stop words** : L'application permet de supprimer les mots les plus fréquents (comme "et", "le", etc.), si nécessaire.
- **Stemming** : Applique différentes techniques de stemming, comme **Porter**, **Lancaster**, et **Snowball**. L'utilisateur peut choisir le type de stemming à appliquer.
- **Visualisation des transformations** : Compare les phrases avant et après le stemming.

### 4. **Options de Configuration dans la Barre Latérale**
Les utilisateurs peuvent personnaliser l'analyse des documents avec les options suivantes dans la barre latérale :

- **Choix de la langue** : Sélectionnez **Français** ou **Anglais** pour le traitement du texte.
- **Descripteur à utiliser** : Choisissez entre **Binaire** ou **Occurrence** pour la représentation des documents.
- **Normalisation** : Choisissez la méthode de normalisation (Aucune, **Probabilité**, ou **L2**).
- **Métrique de distance** : Sélectionnez la métrique de distance à utiliser pour calculer les similarités entre les documents :
  - **Manhattan**
  - **Euclidienne**
  - **Jaccard**
  - **Hamming**
  - **Bray-Curtis**
  - **Kullback-Leibler**
  - **Cosinus**
- **Embeddings** : Choisissez le type d'**embedding** à utiliser :
  - **Word2Vec**
  - **FastText**
  - **Aucun**
- **Nuage de mots** : Affichez un nuage de mots généré à partir du texte ou du fichier téléchargé. Personnalisez la couleur de fond et le nombre de mots à afficher.

### 5. **Méthode d'Entrée du Texte**
Les utilisateurs peuvent choisir comment entrer leur texte dans l'application :
- **Rédaction manuelle** : Saisir ou coller du texte directement dans la zone de texte.
- **Téléchargement de fichier** : Déposer un fichier `.txt` pour l'analyser.

## Fonctionnalités Complètes

### 1. **Analyse de Documents**
- **Chargement de fichiers** : Vous pouvez choisir entre charger un fichier individuel ou un dossier contenant plusieurs fichiers `.txt`.
- **Affichage des phrases détectées** : L'application extrait toutes les phrases des fichiers et les affiche sous forme de liste. Un bouton permet de basculer entre l'affichage et la dissimulation des phrases.
- **Recherche de similarité** : Entrez une phrase pour la comparer aux phrases extraites des documents. Les résultats seront affichés avec des scores de similarité.
- **Nuage de mots** : Un nuage de mots est généré pour le fichier ou le dossier sélectionné.

### 2. **Chatbot**
Le chatbot de ce projet prend en entrée une phrase de l'utilisateur et génère les phrases les plus similaires à partir des documents analysés. Le chatbot fonctionne avec **Cosine Similarity**, et les résultats sont affichés avec leurs scores de similarité.

### 3. **Prétraitement et Stemming**
L'application permet de choisir parmi plusieurs options de prétraitement :
- **Suppression des stop words** : Si activée, cette option supprimera les mots les plus fréquents (en français ou en anglais) avant d'analyser le texte.
- **Stemming** : Applique le stemming avec les algorithmes suivants :
    - **Porter**
    - **Lancaster**
    - **Snowball (Français ou Anglais)**
  
Le stemming peut être appliqué ou non en fonction des besoins. L'application montre les phrases avant et après l'application du stemming, permettant à l'utilisateur de voir les différences.

## Installation

### Prérequis
1. Python 3.x
2. Installez les dépendances nécessaires avec pip.

### Étapes d'installation

1. Clonez ce projet depuis GitHub :
    ```bash
    git clone https://github.com/TBAHRITI-MED/Document-Similarity-Analysis-Tool.git
    ```

2. Accédez au répertoire du projet :
    ```bash
    cd Document-Similarity-Analysis-Tool
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

4. Lancez l'application Streamlit :
    ```bash
    streamlit run streamlit_app.py
    streamlit run recherche_dans_document.py
    streamlit run chatbot.py
    ```

5. Accédez à l'application via votre navigateur à l'adresse suivante :  
   `http://localhost:8501`
   `http://localhost:8502`
   `http://localhost:8503`

## Utilisation

### Page 1 : **Analyse de Documents**
- **Choisir un fichier ou un dossier** : Sélectionnez un dossier contenant des fichiers `.txt` ou un fichier unique à analyser.
- **Affichage des phrases** : Une fois le fichier ou le dossier chargé, les phrases seront extraites et affichées dans l'interface.
- **Recherche de similarité** : Entrez une phrase pour la comparer aux phrases extraites des documents. Les résultats seront affichés avec des scores de similarité.
- **Nuage de mots** : Un nuage de mots est généré pour le fichier ou le dossier sélectionné.

### Page 2 : **Chatbot**
- **Interaction avec le chatbot** : Saisissez une phrase dans l'interface et le chatbot génère une ou plusieurs réponses similaires en fonction des phrases extraites des documents.

### Page 3 : **Prétraitement et Stemming**
- **Suppression des stop words** : Activez cette option pour supprimer les mots fréquents comme "et", "le", etc.
- **Choix du type de stemming** : Sélectionnez le type de stemming à appliquer parmi **Porter**, **Lancaster**, ou **Snowball**.
- **Affichage des résultats** : Comparez les phrases avant et après l'application du stemming.

## Technologies utilisées

- **Python** 3.x
- **Streamlit** : Interface web interactive pour l'analyse de texte.
- **NLTK** : Bibliothèque de traitement du langage naturel pour la tokenisation, le stemming, et la gestion des stop words.
- **Scikit-learn** : Pour le calcul des **TF-IDF** et des **Cosine Similarity**.
- **WordCloud** : Génération de nuages de mots pour visualiser les termes les plus fréquents.
- **Matplotlib & Seaborn** : Pour la création de graphiques et de visualisations.
- **Pandas** : Manipulation des matrices de similarité et des résultats d'analyse.

## Contribuer

Si vous souhaitez contribuer à ce projet, suivez ces étapes :

1. Forkez ce projet.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/nom-de-fonctionnalité`).
3. Commitez vos modifications (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`).
4. Poussez sur la branche (`git push origin feature/nom-de-fonctionnalité`).
5. Ouvrez une pull request.


## Auteur

- **TBAHRITI Mohammed** - Développeur principal

