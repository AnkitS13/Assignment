# Nebula9.ai Support AI

## Overview
This project is designed to function as an AI-powered support agent for **Nebula9.ai**, an industry leader in **Artificial Intelligence, Machine Learning, Cloud Services, and Quality Assurance**. The assistant classifies user queries, generates appropriate responses, and facilitates lead generation while ensuring high-quality, company-specific interactions.

## Features
- **Contextual Question Reformulation**: Reformulates user questions to be standalone without requiring chat history.
- **Intent Classification**: Categorizes user queries into `company_specific`, `general`, or `lead_generation`.
- **Lead Data Collection**: Gathers user details for personalized sales follow-ups.
- **Strict Adherence to Company Branding**: Ensures responses promote **Nebula9.ai** exclusively and avoid misinformation.

## Project Structure
```
|-- prompts.py                # Contains system prompts and classification templates
|-- main.py                   # Core implementation for handling user queries
|-- utils.py                   # Utility functions for data handling
|-- requirements.txt           # Required dependencies
|-- README.md                 # Project documentation (this file)
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnkitS13/Assignment.git
   cd Assignment
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use 'venv\\Scripts\\activate'
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- The assistant processes queries and classifies intent using the `intent_classification_prompt`.
- It uses **LangChain** to manage structured responses and chatbot interactions.
- Lead collection ensures seamless handover for customer engagement.

## Dependencies
- `langchain-core`
- `openai`
- `python-dotenv`


## Screenshot
![Screenshot_Assignment](https://github.com/user-attachments/assets/436b943e-e4bd-40c7-8d87-4cf60e376ff2)
**Fig1**

