import streamlit as st
import os
import asyncio
import logging
import json
import re
from typing import List, Dict, Any

# Database and AI imports
from astrapy.db import AstraDB
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COLLECTION_NAME = "ato_legal_embeddings_hybrid"

# Streamlit config
st.set_page_config(
    page_title="Koala Tax Assistant",
    page_icon="🐨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional CSS
st.markdown("""
<style>
    .main {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Chat message styling */
    .chat-message {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    color: #ffff;
    }
    
    .user-message {
    background-color: #1e1e1e;
    border: 1px solid #383838;
    }
    
    .assistant-message {
    background-color: #1e1e1e;
    border: 1px solid #383838;
    }
    
    /* File note styling */
    .file-note-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #3b82f6;
    }
    
    .section-container {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #383838;
    }
    
    .section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #3b82f6;
    display: flex;
    align-items: center;
    gap: 8px;
    }
    
    .content-text {
    font-size: 0.95rem;
    line-height: 1.5;
    color: #e0e0e0;
    margin: 8px 0;
    }
    
    .key-point {
    background-color: #2a2a2a;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 5px;
    border-left: 3px solid #3b82f6;
    }
    
    .reference-citation {
    font-size: 0.85rem;
    color: #a0a0a0;
    margin-top: 5px;
    }
    
    .source-link {
    color: #3b82f6;
    text-decoration: none;
    }
    
    .source-link:hover {
    text-decoration: underline;
    }
    
    .confidence-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 8px 0;
    }
    
    .confidence-high { background: #166534; color: #dcfce7; }
    .confidence-moderate { background: #92400e; color: #fef3c7; }
    .confidence-low { background: #991b1b; color: #fee2e2; }
    
    .disclaimer {
    font-size: 0.85rem;
    color: #a0a0a0;
    margin-top: 15px;
    padding: 10px;
    border: 1px solid #383838;
    border-radius: 5px;
    background-color: #2a2a2a;
    }
    
    /* Hide Streamlit elements */
    .stChatMessage {
    background-color: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
    background-color: transparent !important;
    padding: 0 !important;
    }
    
    /* Remove extra padding */
    .element-container:empty {
    display: none !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
    }
    
    ::-webkit-scrollbar-track {
    background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
    background: #3b82f6;
    border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
    background: #2563eb;
    }
    
    /* Error message styling */
    .error-message {
    background-color: #2a2a2a;
    border-left: 3px solid #ef4444;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 5px;
    }
    
    .warning-message {
    background-color: #2a2a2a;
    border-left: 3px solid #f59e0b;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_connections():
    """Initializes connections to AstraDB and OpenAI."""
    try:
        astra_db = AstraDB(
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
        )
        collection = astra_db.collection(COLLECTION_NAME)
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        return collection, openai_client
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None, None

async def get_embedding(text: str, client: AsyncOpenAI) -> List[float]:
    """Generates an embedding for a given text using OpenAI."""
    response = await client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
        dimensions=1536
    )
    return response.data[0].embedding

def search_documents(collection, query_embedding: List[float], limit: int = 5) -> List[Dict]:
    """Searches for relevant documents in the collection using a vector."""
    try:
        results = collection.vector_find(
            vector=query_embedding,
            limit=limit,
            fields=["_id", "title", "text_content", "source_info", "document_id", "chunk_order"]
        )
        return list(results)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

async def enhance_query(client: AsyncOpenAI, query: str) -> str:
    """Rephrases a user query to be more specific for searching tax documents."""
    enhancement_prompt = """Rephrase this tax question to be more specific and searchable for Australian taxation documents. 
Focus on key ATO terms and concepts. Keep it concise but precise.

Examples:
"tax rates" → "individual income tax rates Australia 2025-26"
"GST" → "goods and services tax registration requirements"
"super" → "superannuation contribution limits tax deduction"

Return only the enhanced query, no explanation."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": enhancement_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=250,
        temperature=0.1
    )

    enhanced = response.choices[0].message.content.strip()
    logger.info(f"Enhanced query: {query} → {enhanced}")
    return enhanced

async def check_query_intent(client: AsyncOpenAI, query: str) -> bool:
    """Checks if a query is related to Australian taxation."""
    tax_keywords = [
        "tax", "ato", "gst", "income", "deduction", "superannuation", "super", "pillar 2", "pillar two", "IDS", "GloBE", "BEPS",
        "capital gains", "cgt", "fringe benefits", "fbt", "business", "depreciation", "amortisation", "thin capitalisation", "losses",
        "dividend", "offset", "rebate", "lodgment", "return", "assessment", "exemption", "deductions", "audit", "individual",
        "withholding", "payg", "medicare", "levy", "concession", "allowance", "expense", "useful life",
        "claim", "refund", "audit", "ruling", "legislation", "act", "section", "division",
        "resident", "non-resident", "foreign", "trust", "partnership", "company", "sole trader"
    ]
    
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in tax_keywords):
        return True
    
    intent_prompt = """Is this query about Australian taxation or ATO matters? Answer only "yes" or "no".

YES: Australian tax laws, ATO procedures, tax rates, GST, income tax, capital gains, superannuation tax, business tax, tax deductions, tax returns, tax agents, tax exemptions, tax concessions
NO: Financial advice, investment recommendations, non-Australian tax, general chat, personal financial planning that's not tax-related"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": intent_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=5,
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()
    return result == "yes"

def extract_url_from_source(source: str) -> str:
    """Extracts a URL from a source string, handling various formats."""
    if not source:
        return ""
    if source.startswith(('http://', 'https://')):
        return source
    url_match = re.search(r'(https?://[^\s\)\]\,]+)', source)
    if url_match:
        return url_match.group(1)
    if 'ato.gov.au' in source.lower() and not source.startswith('http'):
        clean_source = source.replace('Source: ', '').strip()
        return f"https://www.ato.gov.au/{clean_source}" if not clean_source.startswith('http') else clean_source
    return ""

def extract_title_from_source(source: str) -> str:
    """Extracts a clean title from a source string."""
    url_match = re.search(r'(.*?)[\(\[]https?://.*?[\)\]]', source)
    if url_match:
        return url_match.group(1).strip()
    if source.startswith(('http://', 'https://')):
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', source)
        if domain_match:
            return domain_match.group(1)
    return source

def categorize_sources(context_docs: List[Dict]) -> tuple:
    """Categorizes context documents into legislative and web sources."""
    legislative_sources = []
    web_sources = []

    for doc in context_docs:
        source = doc.get('source_info', '')
        title = doc.get('title', '')
        content = doc.get('text_content', '')

        if 'ACT' in source.upper() and 'SECT' in source.upper():
            legislative_sources.append({'title': title, 'source': source, 'content': content})
        else:
            url = extract_url_from_source(source)
            display_title = title if title and title != 'Unknown' else extract_title_from_source(source)
            web_sources.append({'title': display_title, 'source': source, 'content': content, 'url': url})

    return legislative_sources, web_sources

async def generate_response(client: AsyncOpenAI, query: str, context_docs: List[Dict]) -> str:
    """Generates a response from the AI based on the query and context."""
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        source = doc.get('source_info', 'Unknown')
        title = doc.get('title', 'Unknown')
        content = doc.get('text_content', '')
        url = extract_url_from_source(source)
        context_text += f"Document {i}:\nSource: {source}\nTitle: {title}\nURL: {url}\nContent: {content}\n---\n"

    system_prompt = """You are a professional tax advisor specializing in Australian taxation law. Your task is to provide accurate, specific, and well-structured responses based on the Australian Taxation Office (ATO) website and Australian tax legislation.

CRITICAL INSTRUCTIONS:
1.  Provide SPECIFIC rates, thresholds, and amounts when asked about tax rates. If the question does not contain a year, always use the most recent year's data available in the context.
2.  Include exact figures and percentages from the provided context.
3.  Reference specific legislation sections and ATO guidance documents.
4.  Do not give generic responses - provide the actual data requested.
5.  Provide the response in the style of a professional Australian Tax Law Expert.
6.  Format your response as a professional file note with the sections: Overview, Key Information, Legislation or ATO Reference, Analysis, Conclusion, and Confidence Level.
7.  CRITICAL LINK FORMATTING: When citing a source with a URL, you MUST use the Markdown link format: [Title of the document](URL). Example: [Tax rates for residents](https://www.ato.gov.au/rates/tax-rates-australian-residents). Do not invent URLs or simply write them out.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2100,
        temperature=0.1
    )

    return response.choices[0].message.content

def format_response_as_html(response_text: str, context_docs: List[Dict]) -> str:
    """
    Formats the AI's response text into structured HTML for display,
    preserving the original structure and correctly handling links.
    """
    # --- FIX APPLIED HERE ---
    # First, convert all Markdown-style links `[Text](URL)` to HTML `<a>` tags.
    # This is done before any other processing to ensure all links are correctly formatted.
    processed_text = re.sub(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', r'<a href="\2" target="_blank" class="source-link">\1</a>', response_text)
    
    # Now, we resume with the original formatting logic to preserve the layout.
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_text)
    clean_text = re.sub(r'#{1,6}\s*', '', clean_text)
    
    # Split the response into sections based on the headers, as per the original code.
    sections = re.split(r'\n\s*(?=Overview:|Key Information:|Legislation or ATO Reference:|Analysis:|Conclusion:|Confidence Level:|References:)', clean_text)
    
    html_output = '<div class="file-note-header">📝 File Note</div>'
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        section_parts = section.split(':', 1)
        if len(section_parts) < 2:
            continue
        
        section_title = section_parts[0].strip()
        section_content = section_parts[1].strip()
        
        icon_map = {
            'overview': "📋", 'key information': "📊", 'legislation or ato reference': "⚖️",
            'analysis': "🔍", 'conclusion': "✅", 'confidence level': "🔒", 'references': "📚"
        }
        icon = icon_map.get(section_title.lower().replace(' or ', ' '), "ℹ️")
        
        html_output += f'<div class="section-container"><div class="section-header">{icon} {section_title}</div>'
        
        if 'confidence level' in section_title.lower():
            confidence_text = section_content.lower()
            badge_class = "confidence-moderate"
            if 'high' in confidence_text: badge_class = "confidence-high"
            elif 'low' in confidence_text: badge_class = "confidence-low"
            html_output += f'<div class="confidence-badge {badge_class}">{section_content}</div>'
        
        elif 'references' in section_title.lower():
            # The AI might generate its own reference list; we will ignore it
            # and build our own clean one at the end.
            pass
        else:
            # Process content line by line, preserving paragraphs and bullet points.
            lines = section_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                # The line now contains pre-formatted <a> tags where applicable.
                if line.startswith('* ') or line.startswith('• '):
                    # Strip the bullet character for consistent styling
                    line = re.sub(r'^[*\•]\s*', '', line)
                    html_output += f'<div class="key-point">• {line}</div>'
                else:
                    html_output += f'<div class="content-text">{line}</div>'
        
        html_output += '</div>'
    
    # Add a separate, clean references section at the end.
    legislative_sources, web_sources = categorize_sources(context_docs)
    if legislative_sources or web_sources:
        html_output += '<div class="section-container"><div class="section-header">📚 References</div>'
        seen_sources = set()
        for source in web_sources:
            source_key = source['url'] or source['title']
            if not source_key or source_key in seen_sources: continue
            seen_sources.add(source_key)
            if source['url']:
                html_output += f'<div class="key-point">• <a href="{source["url"]}" target="_blank" class="source-link">{source["title"]}</a></div>'
            else:
                html_output += f'<div class="key-point">• {source["title"]}</div>'
        
        for ref in legislative_sources:
            ref_key = ref['source']
            if not ref_key or ref_key in seen_sources: continue
            seen_sources.add(ref_key)
            html_output += f'<div class="key-point">• {ref["title"]} ({ref["source"]})</div>'
        html_output += '</div>'
    
    html_output += """
    <div class="disclaimer">
    <strong>⚠️ Important Notice:</strong> This information is for general guidance only and is based on current ATO documentation. 
    Tax laws are complex and individual circumstances vary. For personalized advice, please consult a registered tax agent.
    </div>
    """
    
    return html_output

async def process_query(query: str, collection, openai_client):
    """Orchestrates the full query processing pipeline."""
    if not await check_query_intent(openai_client, query):
        return """
        <div class="section-container">
        <div class="section-header">⚠️ Out of Scope Query</div>
        <div class="content-text">
        <p>Koala Tax Assistant can only help with Australian taxation and ATO matters.</p>
        <p>Please ask a question about Australian taxation!</p>
        </div>
        </div>
        """

    try:
        enhanced_query = await enhance_query(openai_client, query)
        query_embedding = await get_embedding(enhanced_query, openai_client)
        relevant_docs = search_documents(collection, query_embedding, limit=5)

        if not relevant_docs:
            return """
            <div class="section-container">
            <div class="section-header">❌ No Information Found</div>
            <div class="content-text">
            <p>No relevant information found in ATO documentation for your query.</p>
            <p><strong>Suggestions:</strong></p>
            <div class="key-point">• Try rephrasing your question with specific tax terms.</div>
            <div class="key-point">• Visit <a href="https://www.ato.gov.au" target="_blank" class="source-link">ato.gov.au</a> for comprehensive information.</div>
            </div>
            </div>
            """

        response_text = await generate_response(openai_client, query, relevant_docs)
        return format_response_as_html(response_text, relevant_docs)

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return f"""
        <div class="section-container">
        <div class="section-header">⚠️ Error</div>
        <div class="content-text">
        <p>An error occurred while processing your query: {str(e)}</p>
        <p>Please try again or contact support if the problem persists.</p>
        </div>
        </div>
        """

def main():
    """Main function to run the Streamlit application."""
    st.markdown("# 🐨 Koala Tax Assistant")
    st.markdown("*Your professional guide to Australian taxation law and ATO guidance*")
    st.markdown(
        """
        **Important Note:** This is a RAG (Retrieval Augmented Generation) application developed using a limited dataset
        sourced from the ATO website and selected Australian tax legislation.
        It may not provide exhaustive answers and should not be considered a substitute for professional
        tax advice. This tool is a proof of concept.
        """)

    collection, openai_client = init_connections()
    if not collection or not openai_client:
        st.error("Failed to initialize backend connections. The assistant is currently unavailable.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("### 💡 Popular Tax Questions")
        col1, col2 = st.columns(2)
        sample_questions = {
            "📊 Individual tax rates 2025-26": "What are the individual income tax rates for 2025-26?",
            "🏢 GST registration requirements": "What are the GST registration requirements for businesses?",
            "🏠 Capital gains tax on property": "How is capital gains tax calculated on investment property?",
            "🏢 Small Business CGT Concessions": "What are the eligibility criteria for the small business CGT concessions?"
        }
        buttons_col1 = list(sample_questions.keys())[:2]
        buttons_col2 = list(sample_questions.keys())[2:]
        
        for col, buttons in zip([col1, col2], [buttons_col1, buttons_col2]):
            with col:
                for button_text in buttons:
                    if st.button(button_text, use_container_width=True):
                        # Using the original rerun logic
                        st.session_state.sample_query = sample_questions[button_text]
                        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "assistant" else "user"):
            # The 'is_html' flag is no longer needed as we always process for HTML
            st.markdown(f'<div class="chat-message {"assistant-message" if message["role"] == "assistant" else "user-message"}">{message["content"]}</div>', unsafe_allow_html=True)

    # Process sample query if it exists (original logic)
    if "sample_query" in st.session_state and st.session_state.sample_query:
        query = st.session_state.pop("sample_query")
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{query}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🐨"):
            with st.spinner("🔍 Researching tax information..."):
                response_html = asyncio.run(process_query(query, collection, openai_client))
                st.markdown(f'<div class="chat-message assistant-message">{response_html}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_html})
        # Rerun to clear the state and wait for next input
        st.rerun()

    # Chat input for new queries
    if prompt := st.chat_input("Ask me about Australian taxation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Rerunning will display the new message and process the response in the next cycle
        st.rerun()

if __name__ == "__main__":
    # This block handles the case where a new message was added and the app was rerun
    if "messages" in st.session_state and st.session_state.messages:
        last_message = st.session_state.messages[-1]
        # Check if the last message is from the user and hasn't been responded to yet
        if last_message["role"] == "user" and len(st.session_state.messages) % 2 != 0:
            collection, openai_client = init_connections()
            if collection and openai_client:
                query = last_message["content"]
                with st.chat_message("assistant", avatar="🐨"):
                    with st.spinner("🔍 Researching tax information..."):
                        response_html = asyncio.run(process_query(query, collection, openai_client))
                        st.markdown(f'<div class="chat-message assistant-message">{response_html}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response_html})

    main()
