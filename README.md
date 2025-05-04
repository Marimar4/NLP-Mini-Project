# NLP-Mini-Project: Sentiment Analysis on Movie Reviews; Improve sentiment classification accuracy with state-of-the art models (embedding or LLMs)



## Introduction

Ce projet explore l’utilisation de méthodes d’embeddings ou de LLMs afin d’améliorer la performance de modèles de classification de texte. Notre objectif est de choisir une représentation adaptée des mots. 
Nous disposons pour cela de bases de données **Open source** sur ensemble de données pour la classification binaire des sentiments qui contient beaucoup plus de données que les ensembles de données de référence précédents. Lequel contient plus de 25 000 critiques de films très polaires pour l'entraînement et 25 000 pour le test. Le but principale de ce traaville est d'améliorer la précision de la classification des sentiments à l'aide de modèles de pointe (intégration ou LLM).

## Revue de litterature

paper: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf


## Objectifs

1. Collecter des données sur les films
2. Nettoyer les données à l’aide d’expressions régulières, puis extraire des tokens qui seront transformés ensuite en lemmes et enfin pondérer ces lemmes avec la méthode TF-IDF (term frequency-inverse document frequency)
3. Visualiser et analyser le jeu de données résultant.
4. Modéliser pour répondre à la problématique grâce aux diverses méthodes de classification et aux modèles de LLM Préentrainer.

**Remarque**

L'approche TF-IDF présente quelques limites : elle ignore l’ordre des mots et produit une représentation éparse, ce qui peut entraîner des problèmes de dimensionnalité et de complexité computationnelle. Pour y remédier, nous pouvons recourir à une représentation plus dense, où chaque mot est projeté dans un vecteur de taille variant librement choisie. Cette méthode, appelée Word2Vec, repose sur l’approche skip-gram et tient compte du contexte local des mots contrairement au LLM tel que BERT qui prend en compte un contexte plus large.



## Bases de données

Les données utilisées dans ce projet proviennent de source suivante : https://ai.stanford.edu/~amaas/data/sentiment/

## Evaluation des résultats

Une fois les mots représentés,  diverses méthodes de classification sont appliqués, telles que Naive Bayes, SVM, la régression logistique ou l’Élastic Net, pour classer les documents. Ainsi que le modèle LLM pré-entraîné BERT afin d’améliorer la performance globale. Pour évaluer les résultats,   la précision, le rappel, le F1-score et l’Accuracy sont mésurées. Les données étant équilibrées, l'on peut se contenter de l’Accuracy pour évaluer la performance de nos modèles.

## Quelques résultats