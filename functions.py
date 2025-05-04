# Importations de la bibliothèque standard

import os
import sys
import re
import io
import json
import random
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter, defaultdict

from numpy import savetxt
import requests
import contractions
import spacy

from gensim.models import Word2Vec
from wordcloud import WordCloud
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.exceptions import ConvergenceWarning
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
nlp = spacy.load("en_core_web_sm")  #modèle anglais, si installé



# Configuration pour limiter les threads et améliorer les performances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Gestionnaire de contexte pour supprimer les avertissements
@contextmanager
def suppress_warnings():
    """
    Supprime divers avertissements pendant l'exécution pour réduire le bruit dans la sortie.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    try:
        yield
    finally:
        warnings.resetwarnings()


#--------------------------
# nettoyage et explorations des textes
#----------------------------------


def clean_texts(text_list):
    html_tag_pattern = re.compile(r'<.*?>')
    cleaned_documents = []

    for text in text_list:
        text = text.lower()
        text = re.sub(html_tag_pattern, ' ', text)
        words = [word.strip() for word in re.split(r"[^a-z]+", text) if word.strip()]
        cleaned_text = ' '.join(words)
        cleaned_documents.append(cleaned_text)
    
    return cleaned_documents



def freq(docs, min_df=20, max_df=2100):
    word_doc_count = defaultdict(set)

    for doc_id, doc in enumerate(docs):
        words = set(doc.split())  # utiliser un set pour éviter les doublons dans un même doc
        for word in words:
            word_doc_count[word].add(doc_id)

    # Filtrer selon le nombre de documents
    filtered = {
        word: len(doc_ids)
        for word, doc_ids in word_doc_count.items()
        if min_df < len(doc_ids) < max_df
    }

    # Optionnel : sauvegarde
    with open('dict.json', 'w') as f:
        json.dump(filtered, f, indent=2)

    return filtered






def process_doc(doc):
    """
    Traite un document (chaîne de caractères) en :
      - Vérifiant si le document contient une contraction mal formée (ex. "don t").
      - Si oui, recolle les tokens ("don" + "t" → "don't") et étend la contraction en "do not".
      - Sinon, retourne le texte tokenisé normalement.
    """
    # Vérifier la présence du motif : mot se terminant par "n", espace, "t"
    if re.sub(r"\b(\w+n)\st\s", r"\1't ", doc, flags=re.IGNORECASE):
        # Recolle les tokens avec une apostrophe : "don t" devient "don't"
        text_fixed = re.sub(r"\b(\w+n)\st\s", r"\1't ", doc, flags=re.IGNORECASE)
        # Appliquer contractions.fix() pour transformer "don't" en "do not"
        text_expanded = contractions.fix(text_fixed)
        return text_expanded.split()
    else:
        # Si le motif n'est pas trouvé, on ne fait rien et on tokenize normalement
        return doc.split()





# Fonctions utilitaires
def extract_texts_from_tar(tar_file, members):
    docs = []
    for member in members:
        f = tar_file.extractfile(member)
        if f:
            text = f.read().decode("utf-8")
            docs.append(text)
    return docs




def word_vectors(neg,pos):
    vectors=[None]*len(neg)+[None]*len(pos)
    for i,doc in enumerate(neg):
        
        words= process_doc(doc)
        vector=[None]*len(words)
        for j,word in enumerate(words):
            vector[j]=word
        vectors[i]=vector
    
    for i,doc in enumerate(pos):
        words= process_doc(doc)
        vector=[None]*len(words)
        for j,word in enumerate(words):
            vector[j]=word
        vectors[i+len(neg)]=vector
    return vectors



         
def pos_tag_docs(docs, batch_size=50, n_process=1):
    """
    docs : liste de documents, 
           chaque document étant une liste de tokens (mots).
    Retourne :
      - pos_counts : un Counter donnant la fréquence de chaque tag (ADJ, NOUN, etc.)
      - tokens_by_pos : un dict de Counters 
                        { 'ADJ': {'great': 10, 'bad': 5, ...}, 'NOUN': {...}, ... }
    """
    pos_counts = Counter()
    tokens_by_pos = defaultdict(Counter)
    
    # Reconstituer tous les textes à partir de la liste de tokens
    texts = [" ".join(token_list) for token_list in docs]
    
    # Utiliser nlp.pipe pour traiter tous les documents en batch
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        for token in doc:
            pos_tag = token.pos_       # ex. "NOUN", "VERB", "ADJ", etc.
            surface_form = token.text  # ou token.lemma_ pour la forme lemmatisée
            pos_counts[pos_tag] += 1
            tokens_by_pos[pos_tag][surface_form.lower()] += 1
            
    return pos_counts, tokens_by_pos

    
def plot_top_tokens_for_tag(tokens_by_pos, tag, top_n=10):
    counter = tokens_by_pos.get(tag, None)
    if not counter:
        print(f"Aucun token trouvé pour le tag {tag}")
        return
    
    most_common = counter.most_common(top_n)
    tokens, freqs = zip(*most_common)
    
    plt.figure(figsize=(8,4))
    plt.barh(tokens, freqs, color='green')
    plt.gca().invert_yaxis()
    plt.xlabel("Fréquence")
    plt.ylabel("Tokens")
    plt.title(f"Top {top_n} tokens pour le tag {tag}")
    plt.show()



# -------------------
# Classes et fonctions de prétraitement
# -------------------
class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=200, n_process=1, excluded_tags=None, model="en_core_web_sm"):
        self.batch_size = batch_size
        self.n_process = n_process
        self.excluded_tags = excluded_tags
        self.model = model
        self.nlp = spacy.load(self.model, disable=["ner", "textcat"])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, (list, tuple)):
            raise ValueError("Input X must be a list or tuple of texts")
        if len(X) == 0:
            return []
        if isinstance(X[0], (list, tuple)):
            docs_text = [" ".join(doc) for doc in X]
        else:
            docs_text = X
        docs_text = [str(doc) for doc in docs_text]
        excluded_tags = set(self.excluded_tags) if self.excluded_tags else set()
        lemmatized_docs = []
        for doc in self.nlp.pipe(
            docs_text,
             batch_size=self.batch_size,
            n_process=self.n_process,
            disable=["ner", "textcat"]
        ):
            lemmatized = " ".join(
                token.lemma_ for token in doc if token.pos_ not in excluded_tags
            )
            lemmatized_docs.append(lemmatized)
        return lemmatized_docs



def lemmatizProcess(docs, excluded_tags=None):
    
    excluded_set = set(excluded_tags) if excluded_tags else set()
    
    # Préallouer le résultat avec la bonne taille
    lemmatized_docs = [None] * len(docs)
    
    # Si les documents sont déjà tokenisés mais nécessitent un traitement spaCy
    docs_text = [" ".join(doc) for doc in docs]
    
    # Traiter les documents avec plus de processus si possible
    for i, doc in enumerate(nlp.pipe(docs_text, batch_size=50, n_process=-1)):
        # Filtrer les tokens dont le pos tag est dans excluded_set
        lemmatized = [token.lemma_ for token in doc if token.pos_ not in excluded_set]
        lemmatized_docs[i] = lemmatized
    
    return lemmatized_docs



# -------------------
# Classes de vectorisation
# -------------------

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=200, window=5, min_count=1, workers=1):
        """
        Initialize Word2Vec vectorizer.

        Parameters:
        - vector_size: Dimensionality of word vectors.
        - window: Maximum distance between current and predicted word.
        - min_count: Ignore words with frequency lower than this.
        - workers: Number of worker threads for training.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        """
        Train Word2Vec model on input texts.

        Parameters:
        - X: List of strings (lemmatized texts).
        """
        # Tokenize each document (split by whitespace since texts are lemmatized)
        tokenized_docs = [doc.split() for doc in X]
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=12345  # For reproducibility
        )
        return self

    def transform(self, X):
        """
        Transform texts into document embeddings by averaging word vectors.

        Parameters:
        - X: List of strings (lemmatized texts).

        Returns:
        - Array of shape (n_samples, vector_size) containing document embeddings.
        """
        tokenized_docs = [doc.split() for doc in X]
        embeddings = []
        for doc in tokenized_docs:
            # Get vectors for words in the document that are in the model's vocabulary
            word_vectors = [self.model.wv[word] for word in doc if word in self.model.wv]
            if word_vectors:
                # Average the word vectors
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                # If no words in vocab, return zero vector
                doc_embedding = np.zeros(self.vector_size)
            embeddings.append(doc_embedding)
        return np.array(embeddings)



















# -------------------
# Étiquetage POS et visualisation
# -------------------

def pos_tag_docs(docs, batch_size=50, n_process=1):
    """
    Effectue l'étiquetage POS sur les documents et calcule les statistiques.

    Paramètres :
    - docs : Liste de documents, chacun étant une liste de tokens.
    - batch_size : Nombre de documents à traiter par lot.
    - n_process : Nombre de processus pour le pipeline spaCy.

    Retourne :
    - pos_counts : Compteur des fréquences des tags POS (par exemple, ADJ, NOUN).
    - tokens_by_pos : Dictionnaire de compteurs associant les tags POS aux fréquences des tokens.
    """
    pos_counts = Counter()
    tokens_by_pos = defaultdict(Counter)
    
    # Reconstitue les textes à partir des listes de tokens
    texts = [" ".join(token_list) for token_list in docs]
    
    # Charge le modèle spaCy si non chargé
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    
    # Traite les documents par lots
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        for token in doc:
            pos_tag = token.pos_
            surface_form = token.text
            pos_counts[pos_tag] += 1
            tokens_by_pos[pos_tag][surface_form.lower()] += 1
            
    return pos_counts, tokens_by_pos

def plot_top_tokens_for_tag(tokens_by_pos, tag, top_n=10):
    """
    Trace les N tokens les plus fréquents pour un tag POS donné.

    Paramètres :
    - tokens_by_pos : Dictionnaire de compteurs issu de pos_tag_docs.
    - tag : Tag POS à tracer (par exemple, 'ADJ', 'NOUN').
    - top_n : Nombre de tokens principaux à afficher.
    """
    import matplotlib.pyplot as plt
    
    counter = tokens_by_pos.get(tag)
    if not counter:
        print(f"Aucun token trouvé pour le tag {tag}")
        return
    
    most_common = counter.most_common(top_n)
    tokens, freqs = zip(*most_common)
    
    plt.figure(figsize=(8, 4))
    plt.barh(tokens, freqs, color='green')
    plt.gca().invert_yaxis()
    plt.xlabel("Fréquence")
    plt.ylabel("Tokens")
    plt.title(f"Top {top_n} Tokens pour le Tag {tag}")
    plt.show()


def document_vector(doc_tokens, model):
    # On récupère les vecteurs pour les tokens présents dans le vocabulaire du modèle
    vectors = [model.wv[token] for token in doc_tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Retourne un vecteur zéro si aucun mot n'est reconnu
        return np.zeros(model.vector_size)
# -------------------
# Fonctions d'évaluation des modèles
# -------------------

def compute_roc(grid_searches, X, y):
    roc_results = {}
    for name, gs in grid_searches.items():
        try:
            # Utiliser le meilleur modèle
            model = gs.best_estimator_
            # Prédire les probabilités
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X)
            else:
                print(f"Le modèle {name} ne supporte ni predict_proba ni decision_function")
                continue
            # Calculer la courbe ROC et l'AUC
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_results[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            }
        except Exception as e:
            print(f"Erreur lors du calcul ROC pour {name}: {str(e)}")
            continue
    return roc_results

def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))









































    