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

# Configuration: Replace or provide these via Streamlit secrets manager.
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

# Professional CSS (style as needed)
st.markdown("""
<style>
.koala-response { font-family: 'Segoe UI', sans-serif; font-size: 1.05rem; }
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
    enhancement_prompt = """Rephrase this tax question to be more specific and searchable for Australian taxation documents. Focus on key ATO terms and concepts. Keep it concise but precise. Examples: "tax rates" → "individual income tax rates Australia 2025-26" "GST" → "goods and services tax registration requirements" "super" → "superannuation contribution limits tax deduction" Return only the enhanced query, no explanation."""
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
        "tax", "ato", "gst", "income", "deduction", "superannuation", "super",
        "pillar 2", "pillar two", "IDS", "GloBE", "BEPS", "capital gains", "cgt",
        "fringe benefits", "fbt", "business", "depreciation", "amortisation",
        "thin capitalisation", "losses", "dividend", "offset", "rebate",
        "lodgment", "return", "assessment", "exemption", "deductions", "audit",
        "individual", "withholding", "payg", "medicare", "levy", "concession",
        "allowance", "expense", "useful life", "claim", "refund", "audit", "ruling",
        "legislation", "act", "section", "division", "resident", "non-resident",
        "foreign", "trust", "partnership", "company", "sole trader"
    ]
    query_lower = query.lower()
    for keyword in tax_keywords:
        if keyword in query_lower:
            return True
    intent_prompt = """Is this query about Australian taxation or ATO matters? Answer only "yes" or "no". YES: Australian tax laws, ATO procedures, tax rates, GST, income tax, capital gains, superannuation tax, business tax, tax deductions, tax returns, tax agents, tax exemptions, tax concessions NO: Financial advice, investment recommendations, non-Australian tax, general chat, personal financial planning that's not tax-related"""
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
    if not source: return ""
    if source.startswith(('http://', 'https://')):
        return source
    url_match = re.search(r'(https?://[^\s\)\]\,]+)', source)
    if url_match:
        return url_match.group(1)
    if 'ato.gov.au' in source.lower() and not source.startswith('http'):
        clean_source = source.replace('Source: ', '').strip()
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
            # Also add variations
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
        context_text += f""" Document {i}: Source: {source} Title: {title} URL: {url} Content: {content} --- """
    system_prompt = """...
You are a professional tax advisor specializing in Australian taxation law. Your task is to provide accurate, specific, and well-structured responses based on the Australian Taxation Office (ATO) website and Australian tax legislation.

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

# ===== PROCESS SOURCE CITES: Only title-as-link, never raw URLs! =====
def process_source_references(text: str, title_to_url: Dict[str, str]) -> str:
    """Convert [source: title] references to [title](url) links if found, else just show the title."""

    def replace_source_ref(match):
        source_title = match.group(1).strip()
        source_url = title_to_url.get(source_title, "")
        if not source_url:
            # Try fuzzy/partial match
            for title, url in title_to_url.items():
                if source_title.lower() in title.lower() or title.lower() in source_title.lower():
                    source_url = url
                    break
        if source_url:
            return f'[{source_title}]({source_url})'
        else:
            return f'{source_title}'

    processed_text = re.sub(r'\[source:\s*([^\]]+)\]', replace_source_ref, text, flags=re.IGNORECASE)
    return processed_text

def format_response_as_html(response_text: str, context_docs: List[Dict]) -> str:
    """Format bot reply for display, replacing source refs to HTML links using titles."""

    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)   # Remove bold
    clean_text = re.sub(r'#{1,6}\s*', '', clean_text)             # Remove headings
    clean_text = re.sub(r'\*\s*', '• ', clean_text)               # Convert bullets

    title_to_url = create_title_url_mapping(context_docs)

    # Convert [source: title](url) to [source: title] to re-link with mapped URLs
    clean_text = re.sub(r'\[source: ([^\]]+)\]\([^)]+\)', r'[source: \1]', clean_text)
    html_body = process_source_references(clean_text, title_to_url)

    # Markdown links to <a href>
    def mdlink_to_html(match):
        text, url = match.groups()
        return f'<a href="{url}" target="_blank">{text}</a>'
    html_body = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', mdlink_to_html, html_body)

    return f'<div class="koala-response">{html_body}</div>'

# ==================
# Streamlit Frontend
# ==================

async def main():
    st.title("🐨 Koala Tax Assistant")
    st.info("Ask your Australian tax question. For sources, only linked titles will display (no naked URLs).")

    # Initialize APIs
    collection, openai_client = init_connections()
    if not collection or not openai_client:
        st.stop()

    user_input = st.text_input("Ask your question about Australian taxation…", key="question")
    do_query = st.button("Ask", use_container_width=True)

    if (user_input.strip() and do_query) or (user_input.strip() and 'auto_query' not in st.session_state):
        with st.spinner("Thinking..."):
            st.session_state['auto_query'] = True
            is_tax = await check_query_intent(openai_client, user_input)
            if not is_tax:
                st.warning("Please ask about Australian tax matters.")
                st.stop()

            enhanced_question = await enhance_query(openai_client, user_input)
            query_embedding = await get_embedding(enhanced_question, openai_client)
            context_docs = search_documents(collection, query_embedding, limit=5)

            bot_reply = await generate_response(openai_client, user_input, context_docs)
            answer_html = format_response_as_html(bot_reply, context_docs)
            st.markdown(answer_html, unsafe_allow_html=True)

            with st.expander("View sources", expanded=False):
                for doc in context_docs:
                    url = extract_url_from_source(doc.get("source_info", ""))
                    title = doc.get("title", "Source")
                    # Only display as a clickable link
                    if url and title:
                        st.markdown(f"- [{title}]({url})")
                    elif title:
                        st.markdown(f"- {title}")

if __name__ == "__main__":
    asyncio.run(main())
