# LLM-model
used to search for the data related to bagavadh gita and patanjali yoga sutras
# SAMAY (Spiritual Assistance and Meditation Aid for You)

SAMAY is a chatbot designed to provide spiritual assistance, meditative practices, and guidance for personal growth. It leverages advanced AI models, including Llama-2, to offer insightful responses based on user queries related to spirituality and meditation.

## Overview

This project includes:
- **A chatbot interface**: Provides spiritual and meditation assistance.
- **Dataset**: The dataset includes spiritual texts and relevant information used to train the system.
- **AI Model**: The system uses the `llama-2-7b-chat` model for generating responses.
- **Retrieval-based system**: It uses a custom-built retrieval system to access specific knowledge from the dataset.

## Features

- **Spiritual Guidance**: Ask the bot questions related to spirituality, philosophy, or self-improvement.
- **Meditation Assistance**: Get personalized meditation practices, including breathing techniques and mindfulness exercises.
- **Self-Reflection**: Prompts and suggestions for personal growth and reflection.

## Setup Instructions

To get started with this project:
1) install the require libraries mentioned in requirements.txt
2) then create a folder name data and then save the datasets which are given in the datasets folder.
3) then run the retriev.py program by using python retriev.py
4) the test the retrieval by running the test_retrieval.py
5) then run the generate.py by **(python generate.py**) and after this run **(chainlit run generate.py)**
