# Smart Study Assistant -- Automatic Question Generator from Study PDFs

## 📌 Project Overview

Smart Study Assistant is a **Machine Learning based NLP system** that
automatically generates exam-style questions from study PDFs.

The system allows users to upload a study-related PDF and automatically
produces: - 2-mark questions - 3-mark questions - 12/16-mark questions

This helps students quickly prepare for exams by identifying important
concepts from study material.

------------------------------------------------------------------------

## 🎯 Objectives

-   Automatically analyze study material from PDFs
-   Extract important concepts using NLP techniques
-   Generate exam-style questions automatically
-   Provide a simple web interface for users

------------------------------------------------------------------------

## 🧠 Is this an LLM?

No.\
This project is **not a Large Language Model (LLM)**.

Instead, it is a **Machine Learning based Natural Language Processing
(NLP) system** that uses classical NLP techniques such as: -
Tokenization - Stopword removal - Lemmatization - Keyword extraction -
Sentence scoring

These techniques are implemented using **NLTK**.

------------------------------------------------------------------------

## ⚙️ Technologies Used

  Technology       Purpose
  ---------------- ----------------------------------
  Python           Core programming language
  Streamlit        Web interface
  NLTK             NLP preprocessing
  PyMuPDF          Extract text from PDF
  Scikit-learn     Machine learning utilities
  Pandas / NumPy   Data processing
  FPDF             Generate downloadable PDF output

------------------------------------------------------------------------

## 🏗️ System Architecture

1.  User uploads a **study PDF**
2.  System extracts text from the PDF
3.  NLP preprocessing is applied
4.  Keywords are extracted
5.  Sentences are ranked based on importance
6.  Questions are generated automatically
7.  Questions can be downloaded as a PDF

------------------------------------------------------------------------

## 🔄 Workflow

1.  Upload PDF
2.  Extract text
3.  Preprocess text using NLP
4.  Identify important keywords
5.  Score sentences
6.  Generate questions
7.  Display and export questions

------------------------------------------------------------------------

## 📂 Project Structure

    question_generator_project
    │
    ├── app.py
    ├── requirements.txt
    ├── nlp_processing
    │   └── preprocess_text.py
    ├── models
    ├── utils
    └── README.md

------------------------------------------------------------------------

## ▶️ How to Run the Project

### Step 1 -- Navigate to project folder

    cd C:\Dev\Study_planner\question_generator_project

### Step 2 -- Activate virtual environment

    venv\Scripts\activate

### Step 3 -- Install dependencies

    pip install -r requirements.txt

### Step 4 -- Run the application

    streamlit run app.py

### Step 5 -- Open browser

    http://localhost:8501

------------------------------------------------------------------------

## 🧪 Example Use Case

A student uploads a **Data Communication and Computer Networks PDF**.\
The system analyzes the document and automatically generates important
exam questions for quick revision.

------------------------------------------------------------------------

## 📊 Advantages

-   Saves study time
-   Automatically identifies important topics
-   Helps in exam preparation
-   Works with any study PDF

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Deep learning based question generation
-   Semantic understanding of text
-   Integration with advanced LLM models
-   Support for multiple languages

------------------------------------------------------------------------



