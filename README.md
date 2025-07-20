# Koala Tax Assistant

Koala Tax Assistant is a **Streamlit-powered Retrieval-Augmented Generation (RAG)** application that provides information on Australian Tax Office (ATO) legal documents. It leverages **AstraDB** for fast vector search and **OpenAI's large language models** for generating relevant and informative responses from ATO tax law snippets.

---

## Features

- **Intelligent Tax Q&A:** Ask questions about Australian tax law and receive answers grounded in official ATO documents.
- **Contextual Retrieval:** Utilizes vector embeddings to find and display the most relevant sections from a knowledge base of ATO documents.
- **AI-Powered Summarization:** OpenAI LLMs generate concise and accurate answers based on the retrieved information.
- **Source Citation:** Provides clickable links to original ATO documents, enabling verification and further reading.
- **Intent Recognition:** Guides users towards appropriate, tax-related queries using basic intent checking.
- **User-Friendly Interface:** Built with Streamlit for an interactive and accessible web application experience.
- **Secure Secret Management:** Integrates with Streamlit's native secrets management to securely handle API keys.

---

## Demo

If you deploy this app on Streamlit Community Cloud, add your app's public URL here.  
For example:  
✨ **Deployed App:** [https://koalabot.streamlit.app]

---

## Technologies Used

- **Python 3.x**
- **Streamlit:** For building the interactive web UI.
- **Astrapy:** Python client for DataStax Astra DB (vector database).
- **OpenAI Python Client:** For accessing GPT and embedding models.
- **DataStax Astra DB:** Vector database used to store and search the ATO document embeddings.
- **OpenAI Embeddings:** For creating vector representations of text.
- **OpenAI GPT Models:** For natural language understanding and generation.

---

## Setup and Installation

Follow these steps to set up and run the Koala Tax Assistant locally or for deployment.

### Prerequisites

- Python 3.8+ installed
- A DataStax Astra DB account with a Vector Database instance
- An OpenAI API key

### 1. Clone the Repository

git clone https://github.com/your_username/koalataxbot.git
cd koalataxbot


### 2. Set up Environment Variables (API Keys)

**Important:** Never hardcode your API keys directly into code or commit them to a public repository.

#### Option A: Using Streamlit Cloud Secrets (Recommended for Deployment)

1. Go to your [Streamlit Community Cloud dashboard](https://share.streamlit.io).
2. Navigate to your app's **Settings → Secrets**.
3. Add your API keys in the following TOML format. Ensure the key names match what your `tax-app.py` expects:

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:your_application_token_here"
ASTRA_DB_API_ENDPOINT = "https://your_api_endpoint_here.datastax.com"
OPENAI_API_KEY = "sk-your_openai_api_key_here"


4. Save the secrets and redeploy your app. `app.py` will access these using `st.secrets["KEY_NAME"]`.

#### Option B: Using a .env File (Recommended for Local Development)

1. Create a file named `.env` in your project root.
2. Add your API keys:

- ASTRA_DB_APPLICATION_TOKEN="AstraCS:your_application_token_here"
- ASTRA_DB_API_ENDPOINT="https://your_api_endpoint_here.datastax.com"
- OPENAI_API_KEY="sk-your_openai_api_key_here"


3. Add `.env` to your `.gitignore` to avoid committing secrets:

- .env


---

### 3. Install Dependencies

It's highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Example `requirements.txt`:

streamlit
astrapy==0.6.0
openai
python-dotenv # Only if using .env locally


---

### 4. Run the Application

Once dependencies are installed and API keys are configured, start the Streamlit app:

streamlit run tax-app.py


This will open the application in your default web browser.

---

## Contributing

*Feel free to contribute! Open pull requests for features, bug fixes, or improvements. Please follow standard best practices for code and documentation.*

---
