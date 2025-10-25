# MedQuery AI

**Build a Complete Medical Intelligent Health Companion using LLMs, LangChain, Pinecone, Streamlit & AWS**

---

## Overview

MedQuery AI is an intelligent medical assistant designed to provide reliable and concise answers to healthcare-related questions.
It uses **Large Language Models (LLMs)** integrated with **LangChain**, **Groq**, and **Pinecone** to process medical documents, generate embeddings, and deliver accurate query-based responses.
This project demonstrates how AI and modern NLP techniques can assist in medical information retrieval, using a lightweight and modular architecture.

---

## Features

* Context-aware medical question answering
* PDF document parsing and knowledge embedding
* Fast semantic search powered by **Pinecone Vector Database**
* Flask backend for query processing
* Streamlit web interface for easy interaction
* Secure environment variable management using `.env`

---

## Project Structure

```
MEDIQUERY_AI/
│
├── data/                     # Data files or PDFs (knowledge source)
├── research/                 # Jupyter notebooks and research experiments
│   └── trials.ipynb
├── src/                      # Core application logic
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── app.py                    # Flask application
├── streamlit.py              # Streamlit web interface
├── store_index.py            # Embedding and Pinecone index creation
├── requirements.txt          # Project dependencies
├── setup.py
├── .env                      # Environment variables
└── README.md
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MedQuery_AI.git
cd MedQuery_AI
```

### 2. Create and activate the environment

```bash
conda create -n medicalai python=3.10 -y
conda activate medicalai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory and add your API keys:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Running the Application

### 1. Store document embeddings to Pinecone

```bash
python store_index.py
```

### 2. Start the backend server

```bash
python app.py
```

### 3. Launch the Streamlit interface

```bash
streamlit run streamlit.py
```

Now, open your browser and navigate to:

```
http://localhost:8501
```

---

## Model & Index Details

* **Groq Model:** `llama-3.3-70b-versatile`
* **Pinecone Index Name:** `medical-queryai`
* **Conda Environment:** `medicalai`

---

## Tech Stack

* **Python**
* **LangChain**
* **Groq LLMs**
* **Pinecone**
* **Streamlit**
* **Flask**
* **AWS**

---
