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
        "tax", "ato", "gst", "income", "deduction", "superannuation", "super", "capital gains", "cgt", 
        "fringe benefits", "fbt", "business", "depreciation", "company", "entertainment", "meal"
    ]
    
    query_lower = query.lower()
    for keyword in tax_keywords:
        if keyword in query_lower:
            return True
    
    intent_prompt = """Is this query about Australian taxation or ATO matters? Answer only "yes" or "no"."""
    
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

        if any(term in source.upper() for term in ['ACT', 'SECT', 'ITAA', 'FBTAA', 'GST']):
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

def create_comprehensive_url_mapping(context_docs: List[Dict]) -> Dict[str, str]:
    """Create comprehensive mapping including legislation and ATO guidance."""
    title_to_url = {}
    
    for doc in context_docs:
        title = doc.get('title', '').strip()
        source = doc.get('source_info', '').strip()
        
        if not title:
            continue
        
        # Extract URL
        url = extract_url_from_source(source)
        
        if url:
            # Add multiple variations for matching
            variations = [title]
            
            # Clean title variations
            clean_title = re.sub(r'^(source:\s*|title:\s*)', '', title, flags=re.IGNORECASE).strip()
            if clean_title != title:
                variations.append(clean_title)
            
            # Handle pipe separators
            if '|' in title:
                variations.append(title.split('|')[0].strip())
            
            # Handle ATO patterns
            if 'Australian Taxation Office' in title:
                variations.append(title.replace('| Australian Taxation Office', '').strip())
            
            # Add all variations
            for variation in variations:
                title_to_url[variation] = url
    
    return title_to_url

def process_all_references(text: str, context_docs: List[Dict]) -> str:
    """Process all references including legislation and ATO guidance."""
    title_to_url = create_comprehensive_url_mapping(context_docs)
    
    def replace_reference(match):
        ref_text = match.group(0)
        
        # Check for section references
        section_match = re.search(r'(Section|Division)\s+([\d\-]+)', ref_text, re.IGNORECASE)
        if section_match:
            section = section_match.group(0)
            # Try to find matching legislation
            for title, url in title_to_url.items():
                if 'ITAA' in title or 'FBTAA' in title or 'GST' in title:
                    return f'<a href="{url}" target="_blank" class="source-link">{ref_text}</a>'
        
        # Check for ATO guidance references
        ato_patterns = ['ATO', 'Australian Taxation Office', 'ato.gov.au']
        for pattern in ato_patterns:
            if pattern.lower() in ref_text.lower():
                for title, url in title_to_url.items():
                    if 'ato' in title.lower():
                        return f'<a href="{url}" target="_blank" class="source-link">{ref_text}</a>'
        
        # Check for exact title matches
        for title, url in title_to_url.items():
            if title.lower() in ref_text.lower() or ref_text.lower() in title.lower():
                return f'<a href="{url}" target="_blank" class="source-link">{ref_text}</a>'
        
        return ref_text
    
    # Process various reference patterns
    text = re.sub(r'(Section|Division)\s+[\d\-]+[\w\(\)]*', replace_reference, text, flags=re.IGNORECASE)
    text = re.sub(r'ATO\s+Guidance:[^<\n]*', replace_reference, text, flags=re.IGNORECASE)
    text = re.sub(r'\[([^\]]+)\]', lambda m: f'[<a href="#" class="source-link">{m.group(1)}</a>]' if m.group(1) in title_to_url else m.group(0), text)
    
    return text

async def generate_response(client: AsyncOpenAI, query: str, context_docs: List[Dict]) -> str:
    context_text = ""
    url_mapping = {}
    
    for i, doc in enumerate(context_docs, 1):
        source = doc.get('source_info', 'Unknown')
        title = doc.get('title', 'Unknown')
        content = doc.get('text_content', '')
        url = extract_url_from_source(source)
        
        if url:
            url_mapping[title] = url
        
        context_text += f"""
Document {i}:
Title: {title}
URL: {url}
Source: {source}
Content: {content}
---
"""

    system_prompt = """You are a professional tax advisor specializing in Australian taxation law. Provide accurate, specific responses.

CRITICAL INSTRUCTIONS:
1. Provide SPECIFIC rates, thresholds, and amounts when asked
2. Include direct URLs to ATO website sections in your response
3. Reference specific legislation sections with links where available
4. Format URLs as clickable links in the text
5. Use this format for references: [ATO Guidance](URL) or [Section 8-1 ITAA 1997](URL)

Format as professional file note:
1. Overview
2. Key Information (with specific data)
3. Legislation/ATO Reference (with embedded URLs)
4. Analysis
5. Conclusion
6. Confidence Level"""

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

def format_response_with_links(response_text: str, context_docs: List[Dict]) -> str:
    """Format response with proper URL linking throughout."""
    
    # Create comprehensive URL mapping
    title_to_url = {}
    for doc in context_docs:
        title = doc.get('title', '').strip()
        source = doc.get('source_info', '').strip()
        url = extract_url_from_source(source)
        
        if url and title:
            title_to_url[title] = url
            
            # Add variations
            clean_title = re.sub(r'^(source:\s*|title:\s*)', '', title, flags=re.IGNORECASE).strip()
            if '|' in clean_title:
                main_title = clean_title.split('|')[0].strip()
                title_to_url[main_title] = url
    
    def replace_with_links(text):
        # Replace [source: title] with clickable links
        def replace_source_ref(match):
            source_title = match.group(1).strip()
            url = title_to_url.get(source_title, "")
            if url:
                return f'<a href="{url}" target="_blank" class="source-link">{source_title}</a>'
            return source_title
        
        text = re.sub(r'\[source:\s*([^\]]+)\]', replace_source_ref, text)
        
        # Replace legislation references
        legislation_patterns = [
            (r'(Income Tax Assessment Act 1997)', 'https://www.legislation.gov.au/Series/C2004A00818'),
            (r'(ITAA 1997)', 'https://www.legislation.gov.au/Series/C2004A00818'),
            (r'(Fringe Benefits Tax Assessment Act 1986)', 'https://www.legislation.gov.au/Series/C2004A01587'),
            (r'(FBTAA 1986)', 'https://www.legislation.gov.au/Series/C2004A01587'),
            (r'(Section 8-1)', 'https://www.legislation.gov.au/Details/C2024C00021/Html/Text#_Toc163895954'),
            (r'(Section 32-5)', 'https://www.legislation.gov.au/Details/C2024C00021/Html/Text#_Toc163896245'),
        ]
        
        for pattern, url in legislation_patterns:
            text = re.sub(pattern, f'<a href="{url}" target="_blank" class="source-link">\\1</a>', text, flags=re.IGNORECASE)
        
        # Replace ATO references
        ato_patterns = [
            (r'ATO Guidance: ([^<\n]*)', 'https://www.ato.gov.au/business/fringe-benefits-tax/'),
            (r'ATO website', 'https://www.ato.gov.au'),
        ]
        
        for pattern, url in ato_patterns:
            text = re.sub(pattern, f'<a href="{url}" target="_blank" class="source-link">\\1</a>', text, flags=re.IGNORECASE)
        
        return text
    
    # Process the entire response
    processed_text = replace_with_links(response_text)
    
    # Split into sections
    sections = re.split(r'\n\s*(?=Overview:|Key Information:|Legislation or ATO Reference:|Analysis:|Conclusion:|Confidence Level:|References:)', processed_text)
    
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
        
        # Set icon
        icon_map = {
            'overview': "📋",
            'key information': "📊",
            'legislation': "⚖️",
            'analysis': "🔍",
            'conclusion': "✅",
            'confidence': "🔒"
        }
        
        icon = next((v for k, v in icon_map.items() if k in section_title.lower()), "ℹ️")
        
        html_output += f'<div class="section-container"><div class="section-header">{icon} {section_title}</div>'
        
        if 'confidence level' in section_title.lower():
            confidence_text = section_content.lower()
            badge_class = "confidence-high" if 'high' in confidence_text else "confidence-moderate" if 'moderate' in confidence_text else "confidence-low"
            html_output += f'<div class="confidence-badge {badge_class}">{section_content}</div>'
        else:
            lines = section_content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('•'):
                        html_output += f'<div class="key-point">{line}</div>'
                    else:
                        html_output += f'<div class="content-text">{line}</div>'
        
        html_output += '</div>'
    
    return html_output

async def process_query(query: str, collection, openai_client):
    is_tax_query = await check_query_intent(openai_client, query)

    if not is_tax_query:
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
            <p>No relevant information found in ATO documentation</p>
            </div>
            </div>
            """

        response_text = await generate_response(openai_client, query, relevant_docs)
        return format_response_with_links(response_text, relevant_docs)

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return f"""
        <div class="section-container">
        <div class="section-header">⚠️ Error</div>
        <div class="content-text">
        <p>An error occurred while processing your query: {str(e)}</p>
        </div>
        </div>
        """

def main():
    st.markdown("# 🐨 Koala Tax Assistant")
    st.markdown("*Your professional guide to Australian taxation law and ATO guidance*")

    collection, openai_client = init_connections()
    if not collection or not openai_client:
        st.error("Failed to initialize connections. Please check your configuration.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🐨"):
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me about Australian taxation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{prompt}</div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="🐨"):
            with st.spinner("🔍 Researching tax information..."):
                response_html = asyncio.run(process_query(prompt, collection, openai_client))
                st.markdown(response_html, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_html})

if __name__ == "__main__":
    main()
