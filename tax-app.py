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
    response = await client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
        dimensions=1536
    )
    return response.data[0].embedding

def search_documents(collection, query_embedding: List[float], limit: int = 5) -> List[Dict]:
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
    tax_keywords = [
        "tax", "ato", "gst", "income", "deduction", "superannuation", "super", "pillar 2", "pillar two", "IDS", "GloBE", "BEPS",
        "capital gains", "cgt", "fringe benefits", "fbt", "business", "depreciation", "amortisation", "thin capitalisation", "losses",
        "dividend", "offset", "rebate", "lodgment", "return", "assessment", "exemption", "deductions", "audit", "individual",
        "withholding", "payg", "medicare", "levy", "concession", "allowance", "expense", "useful life",
        "claim", "refund", "audit", "ruling", "legislation", "act", "section", "division",
        "resident", "non-resident", "foreign", "trust", "partnership", "company", "sole trader"
    ]
    
    query_lower = query.lower()
    for keyword in tax_keywords:
        if keyword in query_lower:
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
        max_tokens=150,
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()
    return result == "yes"

def extract_url_from_source(source: str) -> str:
    if not source:
        return ""

    if source.startswith(('http://', 'https://')):
        return source

    url_match = re.search(r'(https?://[^\s\)\]\,]+)', source)
    if url_match:
        return url_match.group(1)

    if 'ato.gov.au' in source.lower() and not source.startswith('http'):
        clean_source = source.replace('Source: ', '').strip()
        if not clean_source.startswith('http'):
            return f"https://www.ato.gov.au/{clean_source}"

    return ""

def extract_title_from_source(source: str) -> str:
    url_match = re.search(r'(.*?)[\(\[]https?://.*?[\)\]]', source)
    if url_match:
        return url_match.group(1).strip()
    
    if source.startswith(('http://', 'https://')):
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', source)
        if domain_match:
            return domain_match.group(1)
    
    return source

def categorize_sources(context_docs: List[Dict]) -> tuple:
    legislative_sources = []
    web_sources = []

    for doc in context_docs:
        source = doc.get('source_info', '')
        title = doc.get('title', '')
        content = doc.get('text_content', '')

        if 'ACT' in source.upper() and 'SECT' in source.upper():
            legislative_sources.append({
                'title': title,
                'source': source,
                'content': content
            })
        else:
            url = extract_url_from_source(source)
            display_title = title if title and title != 'Unknown' else extract_title_from_source(source)
            
            web_sources.append({
                'title': display_title,
                'source': source,
                'content': content,
                'url': url
            })

    return legislative_sources, web_sources

def create_title_url_mapping(context_docs: List[Dict]) -> Dict[str, str]:
    title_to_url = {}
    
    for doc in context_docs:
        title = doc.get('title', '').strip()
        source = doc.get('source_info', '').strip()
        
        if not title or title == 'Unknown':
            continue
        
        url = extract_url_from_source(source)
        
        if url:
            title_to_url[title] = url
            
            clean_title = re.sub(r'^(source:\s*|title:\s*)', '', title, flags=re.IGNORECASE).strip()
            if clean_title != title:
                title_to_url[clean_title] = url
            
            if '|' in title:
                main_title = title.split('|')[0].strip()
                title_to_url[main_title] = url
            
            if 'Australian Taxation Office' in title:
                short_title = title.replace('| Australian Taxation Office', '').strip()
                title_to_url[short_title] = url
    
    return title_to_url

async def generate_response(client: AsyncOpenAI, query: str, context_docs: List[Dict]) -> str:
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        source = doc.get('source_info', 'Unknown')
        title = doc.get('title', 'Unknown')
        content = doc.get('text_content', '')
        url = extract_url_from_source(source)

        context_text += f"""
Document {i}:
Source: {source}
Title: {title}
URL: {url}
Content: {content}
---
"""

    system_prompt = """You are a professional tax advisor specializing in Australian taxation law. Your task is to provide accurate, specific, and well-structured responses based on the Australian Taxation Office (ATO) website and Australian tax legislation.

CRITICAL INSTRUCTIONS:
1. Provide SPECIFIC rates, thresholds, and amounts when asked about tax rates and if the question does not contain any year, always consider the most latest year
2. Include exact figures and percentages from the provided context and always use the latest year figures when not asked specifically
3. Reference specific legislation sections and ATO guidance documents
4. Include direct URLs to ATO website sections only when available and make sure to link the URL to the title and DO NOT show full URL
5. Do not give generic responses - provide the actual data requested
6. Provide response like a professional Australian Tax Law Expert
7. IMPORTANT! always link the URL with the title when the URL is available

Format your response as a professional file note with the following sections:
1. Overview: A concise summary of the query and main findings (2-3 sentences)
2. Key Information: The most important points with specific rates, amounts, and thresholds
3. Legislation or ATO Reference: Specific sections of legislation or ATO guidance with URLs and if URLs are not availble, provide specific referece to the section
4. Analysis: Your professional interpretation of how the law applies or what is the interpretation of the law
5. Conclusion: A clear summary of the answer
6. Confidence Level: High/Moderate/Low with explanation

IMPORTANT: When asked about tax rates, provide the actual rates and thresholds, not general statements about where to find them."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2000,
        temperature=0.05
    )

    return response.choices[0].message.content

def process_source_references(text: str, title_to_url: Dict[str, str]) -> str:
    """Process a line of text to convert [source: title], [Title](URL), or raw URL references to clickable links."""
    
    # First: Convert markdown links [Title](URL) to [source: Title]
    def replace_markdown_link(match):
        title = match.group(1).strip()
        url = match.group(2).strip()
        # Store in mapping if not present
        if title and url:
            if title not in title_to_url:
                title_to_url[title] = url
        return f'[source: {title}]'
    
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', replace_markdown_link, text)

    # Second: Replace [source: title] with clickable links
    def replace_source_ref(match):
        source_title = match.group(1).strip()
        source_url = title_to_url.get(source_title, "")
        
        if not source_url:
            for title, url in title_to_url.items():
                if source_title.lower() in title.lower() or title.lower() in source_title.lower():
                    source_url = url
                    break
        
        if source_url:
            return f'[source: <a href="{source_url}" target="_blank" class="source-link">{source_title}</a>]'
        else:
            return f'[source: {source_title}]'
    
    text = re.sub(r'\[source:\s*([^\]]+)\]', replace_source_ref, text)

    # Third: Handle any remaining standalone URLs that follow a title pattern
    def replace_standalone_url(match):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if title and url:
            if title not in title_to_url:
                title_to_url[title] = url
            return f'<a href="{url}" target="_blank" class="source-link">{title}</a>'
        return match.group(0)
    
    text = re.sub(r'([A-Za-z0-9\s\-\–\—]+)\s*\(?(https?://[^\s\)\]\,]+)\)?', replace_standalone_url, text)
    
    return text

def format_response_as_html(response_text: str, context_docs: List[Dict]) -> str:
    """Format the response as HTML to be displayed in the chat interface."""
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    clean_text = re.sub(r'#{1,6}\s*', '', clean_text)
    clean_text = re.sub(r'\*\s*', '• ', clean_text)

    title_to_url = create_title_url_mapping(context_docs)

    # Process markdown links and raw URLs before sectioning
    clean_text = process_source_references(clean_text, title_to_url)

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
        
        if 'overview' in section_title.lower():
            icon = "📋"
        elif 'key information' in section_title.lower():
            icon = "📊"
        elif 'legislation' in section_title.lower() or 'ato reference' in section_title.lower():
            icon = "⚖️"
        elif 'analysis' in section_title.lower():
            icon = "🔍"
        elif 'conclusion' in section_title.lower():
            icon = "✅"
        elif 'confidence' in section_title.lower():
            icon = "🔒"
        elif 'references' in section_title.lower():
            icon = "📚"
        else:
            icon = "ℹ️"
        
        html_output += f'<div class="section-container"><div class="section-header">{icon} {section_title}</div>'
        
        if 'confidence level' in section_title.lower():
            confidence_text = section_content.lower()
            badge_class = "confidence-high" if 'high' in confidence_text else "confidence-moderate" if 'moderate' in confidence_text else "confidence-low"
            html_output += f'<div class="confidence-badge {badge_class}">{section_content}</div>'
        elif 'references' in section_title.lower():
            pass
        else:
            lines = section_content.split('\n')
            for line in lines:
                line = line.strip()
                if line and line.startswith('•'):
                    processed_line = line  # Already processed above
                    html_output += f'<div class="key-point">{processed_line}</div>'
                elif line:
                    processed_line = line  # Already processed
                    html_output += f'<div class="content-text">{processed_line}</div>'
        
        html_output += '</div>'
    
    legislative_sources, web_sources = categorize_sources(context_docs)
    
    html_output += '<div class="section-container"><div class="section-header">📚 References</div>'
    
    seen_sources = set()
    for source in web_sources:
        source_key = f"{source['source']}_{source['title']}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        
        if source['url']:
            html_output += f'<div class="key-point">• <a href="{source["url"]}" target="_blank" class="source-link">{source["title"]}</a></div>'
        else:
            html_output += f'<div class="key-point">• {source["title"]}</div>'
    
    for ref in legislative_sources:
        ref_key = f"{ref['source']}_{ref['title']}"
        if ref_key in seen_sources:
            continue
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
    is_tax_query = await check_query_intent(openai_client, query)

    if not is_tax_query:
        return """
        <div class="section-container">
        <div class="section-header">⚠️ Out of Scope Query</div>
        <div class="content-text">
        <p>Koala Tax Assistant can only help with Australian taxation and ATO matters.</p>
        <p><strong>I can help with:</strong></p>
        <div class="key-point">• Australian tax laws and regulations</div>
        <div class="key-point">• Tax returns and deductions</div>
        <div class="key-point">• GST and income tax questions</div>
        <div class="key-point">• Superannuation tax matters</div>
        <div class="key-point">• Business tax obligations</div>
        <div class="key-point">• Tax agent services</div>
        </div>
        <div class="content-text">
        <p><strong>I cannot help with:</strong></p>
        <div class="key-point">• Financial advice or investment recommendations</div>
        <div class="key-point">• Non-Australian tax matters</div>
        <div class="key-point">• General financial planning</div>
        <div class="key-point">• Personal financial decisions</div>
        </div>
        <div class="content-text">
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
            <p>No relevant information found in ATO documentation</p>
            <p><strong>Suggestions:</strong></p>
            <div class="key-point">• Try rephrasing your question with specific tax terms</div>
            <div class="key-point">• Contact the ATO directly on <strong>13 28 61</strong></div>
            <div class="key-point">• Visit <a href="https://www.ato.gov.au" target="_blank" class="source-link">ato.gov.au</a> for comprehensive information</div>
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
        <p>Please try again or contact the ATO directly for assistance.</p>
        </div>
        </div>
        """

def main():
    st.markdown("# 🐨 Koala Tax Assistant")
    st.markdown("*Your professional guide to Australian taxation law and ATO guidance*")
    st.markdown(
    """
    **Important Note:** This is a RAG (Retrieval Augmented Generation) application developed using a limited dataset
    sourced from the ATO website and selected Australian tax legislation (including relevant sections of the Corporations Act).
    As such, it may not provide exhaustive or fully comprehensive answers, nor should it be considered a substitute for professional
    tax advice. This tool may also not perform accurate calculations. It is currently a proof of concept.
    """)

    collection, openai_client = init_connections()
    if not collection or not openai_client:
        st.error("Failed to initialize connections. Please check your configuration.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("### 💡 Popular Tax Questions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Individual tax rates 2025-26", use_container_width=True):
                st.session_state.sample_query = "What are the individual income tax rates for 2025-26?"
                st.rerun()
            if st.button("🏢 GST registration requirements", use_container_width=True):
                st.session_state.sample_query = "What are the GST registration requirements for businesses?"
                st.rerun()

        with col2:
            if st.button("🏠 Capital gains tax on property", use_container_width=True):
                st.session_state.sample_query = "How is capital gains tax calculated on investment property?"
                st.rerun()
            if st.button("🏢 Small Business CGT Concessions", use_container_width=True):
                st.session_state.sample_query = "What are the eligibility criteria for the small business CGT concessions in Australia for 2025-26, and how do they reduce capital gains?"
                st.rerun()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🐨"):
                if message.get("is_html", False):
                    st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    if hasattr(st.session_state, 'sample_query'):
        query = st.session_state.sample_query
        del st.session_state.sample_query

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{query}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🐨"):
            with st.spinner("🔍 Researching tax information..."):
                response_html = asyncio.run(process_query(query, collection, openai_client))
            st.markdown(f'<div class="chat-message assistant-message">{response_html}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_html, "is_html": True})

    if prompt := st.chat_input("Ask me about Australian taxation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🐨"):
            with st.spinner("🔍 Researching tax information..."):
                response_html = asyncio.run(process_query(prompt, collection, openai_client))
            st.markdown(f'<div class="chat-message assistant-message">{response_html}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_html, "is_html": True})

if __name__ == "__main__":
    main()
