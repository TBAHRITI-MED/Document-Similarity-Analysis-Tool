import streamlit as st
import chardet
import nltk
nltk.download("punkt", quiet=True)  # Télécharge le tokenizer NLTK
from nltk.tokenize import sent_tokenize
import numpy as np

# Pour l'approche basique (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pour l'approche avancée Sentence-BERT
from sentence_transformers import SentenceTransformer, util

# Pour l'approche avancée QA
from transformers import pipeline, CamembertTokenizer

##############################################
# 1) Chargement / cache des modèles
##############################################
@st.cache_resource
def load_retrieval_model():
    """
    Modèle Sentence-BERT multilingue pour la recherche sémantique.
    """
    return SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

@st.cache_resource
def load_qa_pipeline():
    """
    Pipeline QA francophone (CamemBERT fine-tuné sur FQuAD / Piaf),
    en forçant l'usage du tokenizer lent.
    """
    my_tokenizer = CamembertTokenizer.from_pretrained(
        "etalab-ia/camembert-base-squadFR-fquad-piaf",
        use_fast=False,           
        trust_remote_code=True
    )
    my_pipeline = pipeline(
        "question-answering",
        model="etalab-ia/camembert-base-squadFR-fquad-piaf",
        tokenizer=my_tokenizer
    )
    return my_pipeline

retrieval_model = load_retrieval_model()
qa_pipeline = load_qa_pipeline()

###########################################
# 2) Mise en forme Streamlit
###########################################
# Petit style d'accueil
st.markdown("""
<style>
.big-title {
    font-size:250%;
    color: #2F4F4F;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🤖 Chatbot : Basique ou Avancé</p>', unsafe_allow_html=True)

# Description
st.write("**Bienvenue sur ce chatbot**. Sélectionnez l'approche (basique TF-IDF amélioré ou avancée Sentence-BERT + QA), et choisissez si vous souhaitez charger un seul fichier ou plusieurs.")

###########################################
# 3) Paramètres / Sélection
###########################################
approach = st.selectbox(
    "🤔 Choisissez l'approche souhaitée :",
    ["Basique (TF-IDF amélioré)", "Avancée (Hugging Face QA)"]
)

mode_fichiers = st.radio(
    "📂 Souhaitez-vous charger :",
    ("Un seul fichier", "Plusieurs fichiers")
)

###########################################
# 4) Fonctions Lecture + Décodage
###########################################
def lire_fichier_streamlit(upload):
    raw_data = upload.read()
    detect_result = chardet.detect(raw_data)
    guessed_encoding = detect_result["encoding"] if detect_result["encoding"] else "utf-8"
    document = raw_data.decode(guessed_encoding, errors="replace")
    st.write(f"**Encodage détecté** : {guessed_encoding}")
    return document

###########################################
# 5) Logique "Un Seul Fichier"
###########################################
if mode_fichiers == "Un seul fichier":
    uploaded_file = st.file_uploader("🔎 Téléchargez un fichier .txt", type="txt")
    if uploaded_file is not None:
        doc_str = lire_fichier_streamlit(uploaded_file)
        # Découpage en phrases
        sentences = sent_tokenize(doc_str)
        st.write("### 📜 Phrases détectées :")
        for idx, sentence in enumerate(sentences, start=1):
            st.write(f"{idx}. {sentence}")

        # 5A) Approche Basique TF-IDF amélioré
        if approach == "Basique (TF-IDF amélioré)":
            st.markdown("#### 🏷️ Basique Amélioré : TF-IDF (n-grams, re-ranking)")

            user_input = st.text_input("💬 Posez votre question :")
            top_k = st.slider("Nombre de phrases à présélectionner ?", 1, 10, 3)

            if user_input:
                # Vectorisation TF-IDF (n-grams, pas de stop_words, max_df=0.8)
                vectorizer = TfidfVectorizer(
                    ngram_range=(1,2),
                    max_df=0.8
                )
                corpus = [user_input] + sentences
                tfidf_matrix = vectorizer.fit_transform(corpus)
                question_tfidf = tfidf_matrix[0:1]
                corpus_tfidf = tfidf_matrix[1:]

                sims = cosine_similarity(question_tfidf, corpus_tfidf).flatten()
                # On récupère top_k
                sorted_idx = np.argsort(sims)[::-1]
                top_idx = sorted_idx[:top_k]

                # Mot le plus significatif
                q_vec = question_tfidf.toarray().flatten()
                feature_names = vectorizer.get_feature_names_out()
                if np.sum(q_vec) != 0:
                    max_i = np.argmax(q_vec)
                    most_sig_token = feature_names[max_i]
                else:
                    most_sig_token = ""

                # Re-ranking
                re_rank = []
                for idx_i in top_idx:
                    base_score = sims[idx_i]
                    bonus = 0.0
                    if most_sig_token and most_sig_token.lower() in sentences[idx_i].lower():
                        bonus = 0.1
                    new_score = base_score + bonus
                    re_rank.append((idx_i, new_score))

                re_rank.sort(key=lambda x: x[1], reverse=True)
                best_idx = re_rank[0][0]
                best_score = re_rank[0][1]
                best_sentence = sentences[best_idx]

                # Politesse
                polite_prefix = ""
                question_lower = user_input.lower().strip()
                if question_lower.startswith("comment"):
                    polite_prefix = "Après analyse, "
                elif question_lower.startswith("pourquoi"):
                    polite_prefix = "Car, "
                elif question_lower.startswith("peux-tu"):
                    polite_prefix = "Oui, bien sûr! "

                # Affichage
                st.write("**Réponse :**")
                st.write(polite_prefix + best_sentence)
                st.write("---")
                st.write(f"**Phrase la plus pertinente** (score={best_score:.4f}) : {best_sentence}")
                if most_sig_token:
                    st.write(f"(mot-clé le plus important : **{most_sig_token}**)")

        # 5B) Approche Avancée
        else:
            st.markdown("#### 🚀 Avancée : Sentence-BERT + QA CamemBERT")
            k = st.slider("Combien de phrases combiner pour le contexte ?", 1, 10, 3)
            user_question = st.text_input("💬 Posez votre question :")

            if user_question:
                st.write("🔎 Calcul de la similarité sémantique via Sentence-BERT...")
                embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)
                question_emb = retrieval_model.encode(user_question, convert_to_tensor=True)
                sims = util.cos_sim(question_emb, embeddings)[0].cpu().numpy()

                sorted_indices = np.argsort(sims)[::-1]
                top_k_sentences = [sentences[i] for i in sorted_indices[:k]]
                context = " ".join(top_k_sentences)

                st.write("**Phrases sélectionnées :**")
                for i, idx in enumerate(sorted_indices[:k], start=1):
                    st.write(f"{i}) [score={sims[idx]:.4f}] : {sentences[idx]}")

                qa_input = {"question": user_question, "context": context}
                result = qa_pipeline(qa_input)
                st.write("**Réponse générée** : ", result["answer"])
                st.write(f"(Confiance : {result['score']:.4f})")
    else:
        st.info("Veuillez téléverser un fichier .txt pour commencer.")

###########################################
# 6) Logique "Plusieurs Fichiers"
###########################################
else:
    uploaded_files = st.file_uploader("📂 Téléchargez un ou plusieurs fichiers .txt", type="txt", accept_multiple_files=True)

    if uploaded_files:
        all_sentences = []
        for file_item in uploaded_files:
            doc = lire_fichier_streamlit(file_item)
            segs = sent_tokenize(doc)
            all_sentences.extend(segs)

        st.success(f"{len(uploaded_files)} fichier(s) téléversé(s), total de {len(all_sentences)} phrases.")
        st.write("**Extrait de phrases** :")
        for i, s in enumerate(all_sentences[:12], start=1):
            st.write(f"{i}. {s}")

        if approach == "Basique (TF-IDF amélioré)":
            st.markdown("#### 🏷️ Basique Amélioré : TF-IDF (n-grams, re-ranking)")
            user_input = st.text_input("💬 Entrez votre question :")
            top_k = st.slider("Nombre de phrases à présélectionner ?", 1, 10, 3)

            if user_input:
                # n-grams, pas de stopwords, max_df=0.8
                vectorizer = TfidfVectorizer(
                    ngram_range=(1,2),
                    max_df=0.8
                )
                corpus = [user_input] + all_sentences
                tfidf_mat = vectorizer.fit_transform(corpus)
                question_tf = tfidf_mat[0:1]
                corpus_tf = tfidf_mat[1:]
                sims = cosine_similarity(question_tf, corpus_tf).flatten()

                # On prend top_k
                sorted_ids = np.argsort(sims)[::-1]
                top_ids = sorted_ids[:top_k]

                # Mot-clé le plus important
                q_vec = question_tf.toarray().flatten()
                feat_names = vectorizer.get_feature_names_out()
                if np.sum(q_vec) != 0:
                    max_i = np.argmax(q_vec)
                    most_sig = feat_names[max_i]
                else:
                    most_sig = ""

                # Re-rank
                re_rk = []
                for idx_i in top_ids:
                    base_score = sims[idx_i]
                    bonus = 0.0
                    if most_sig and most_sig.lower() in all_sentences[idx_i].lower():
                        bonus += 0.1
                    new_score = base_score + bonus
                    re_rk.append((idx_i, new_score))

                re_rk.sort(key=lambda x: x[1], reverse=True)
                best_idx = re_rk[0][0]
                best_score = re_rk[0][1]
                best_sentence = all_sentences[best_idx]

                # Politesse
                prefix = ""
                q_low = user_input.lower().strip()
                if q_low.startswith("comment"):
                    prefix = "Après analyse, "
                elif q_low.startswith("pourquoi"):
                    prefix = "Car, "
                elif q_low.startswith("peux-tu"):
                    prefix = "Oui, bien sûr! "

                st.write("**Réponse :**")
                st.write(prefix + best_sentence)
                st.write("---")
                st.write(f"**Phrase la plus pertinente** (score={best_score:.4f}) : {best_sentence}")
                if most_sig:
                    st.write(f"(mot-clé le plus important : **{most_sig}**)")

        else:
            st.markdown("#### 🚀 Avancée : Sentence-BERT + QA CamemBERT")
            k = st.slider("Combien de phrases combiner pour le contexte ?", 1, 10, 3)
            user_q = st.text_input("💬 Posez votre question :")
            if user_q:
                st.write("🔎 Calcul de la similarité via Sentence-BERT...")

                embeddings = retrieval_model.encode(all_sentences, convert_to_tensor=True)
                q_emb = retrieval_model.encode(user_q, convert_to_tensor=True)
                sims = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()

                sorted_i = np.argsort(sims)[::-1]
                top_sents = [all_sentences[i] for i in sorted_i[:k]]
                ctx = " ".join(top_sents)

                st.write("**Phrases sélectionnées :**")
                for i, idx in enumerate(sorted_i[:k], start=1):
                    st.write(f"{i}) [score={sims[idx]:.4f}] : {all_sentences[idx]}")

                qa_input = {"question": user_q, "context": ctx}
                res = qa_pipeline(qa_input)
                st.write("**Réponse générée** : ", res["answer"])
                st.write(f"(Confiance : {res['score']:.4f})")

    else:
        st.info("Veuillez téléverser au moins un fichier .txt pour commencer.")
