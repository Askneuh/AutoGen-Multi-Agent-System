# 🤖 AutoGen Research & Professor Pipeline

Ce projet implémente un pipeline multi-agent autonome capable de rechercher des informations techniques sur le Web, de rédiger des tutoriels structurés et de servir de base de connaissances interactive grâce au RAG (Retrieval-Augmented Generation).

## 🌟 Fonctionnalités

* **Recherche Web Intelligente** : Utilise l'API Tavily pour extraire du contenu pertinent en ignorant les sources non textuelles comme les réseaux sociaux.
* **Équipe de Rédaction Collaborative** : Quatre agents (Scout, Reader, Writer, Critic) collaborent via une `SelectorGroupChat` pour produire un contenu de haute qualité.
* **Boucle de Critique Technique** : Un agent `Critic` évalue le tutoriel sur 20 points et force une révision si le score est inférieur à 15/20.
* **Mémoire Vectorielle (RAG)** : Indexation automatique du tutoriel final dans **ChromaDB** pour une interrogation ultérieure.
* **Mode Professeur** : Un mode interactif où un agent dédié répond à vos questions en se basant exclusivement sur la base de connaissances générée.

## 🏗️ Architecture des Agents

Le projet s'appuie sur le framework **AutoGen** pour orchestrer les rôles suivants :

| Agent | Modèle LLM | Rôle Principal |
| :--- | :--- | :--- |
| **Scout** | Llama 3.1 8B | Trouve des URLs pertinentes sur le web. |
| **Reader** | Llama 3.1 8B | Analyse et synthétise techniquement le contenu des pages. |
| **Writer** | Llama 3.3 70B | Rédige le tutoriel final en Markdown avec du code Python. |
| **Critic** | Llama 3.3 70B | Valide la qualité technique et la complétude du texte. |
| **Professor**| Llama 3.3 70B | Enseigne le sujet en interrogeant la base vectorielle. |

## 🛠️ Installation & Configuration

### 1. Dépendances
Installez les bibliothèques nécessaires :
```bash
pip install autogen-agentchat chromadb sentence-transformers tavily-python rich
