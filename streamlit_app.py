# RAG_v3/streamlit_app.py
import os
import sys

# Ensure UTF-8 encoding for Windows to handle Unicode characters
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except (AttributeError, TypeError):
        # If stdout/stderr don't have buffer attribute, skip
        pass

# SQLite3 fix for Streamlit Cloud (must be before any Chroma imports)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

# Set Streamlit Cloud environment flag if running on cloud
if any(env in os.environ for env in ["STREAMLIT_SHARING_MODE", "STREAMLIT_RUNTIME_ENV"]):
    os.environ["STREAMLIT_CLOUD"] = "true"
    
# Note: Chroma configuration now handled in utils/chroma_utils.py
# to use the new client API with proper migration handling

from pathlib import Path
import base64
import streamlit as st

# ---------- SETTINGS ----------
st.set_page_config(
    page_title="Knowledge Assistant", 
    layout="wide", 
    page_icon="🔍",
    initial_sidebar_state="expanded"
)
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
VDB_DIR = ROOT / "vectordb"

# Use temporary directory on Streamlit Cloud for uploads
if os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser"):
    # On Streamlit Cloud, use temporary directory
    import tempfile
    temp_base = Path(os.environ.get("TMPDIR", "/tmp"))
    NEW_DOC_DIR = temp_base / "new_uploads"
else:
    NEW_DOC_DIR = ROOT / "data" / "new_uploads"  # Separate directory for new document uploads

DATA_DIR.mkdir(parents=True, exist_ok=True)
VDB_DIR.mkdir(parents=True, exist_ok=True)
NEW_DOC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ENV / AUTH ----------
# Import secrets manager
from utils.secrets_manager import get_auth_config, set_environment_from_secrets

# Set environment variables from Streamlit secrets
set_environment_from_secrets()

# Get authentication config
auth_config = get_auth_config()
USERNAME = auth_config["username"]
PASSWORD = auth_config["password"]

# Initialize session state
if "auth" not in st.session_state: 
    st.session_state.auth = False
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
if "rag_option" not in st.session_state: 
    st.session_state.rag_option = "Current documents"
if "temperature" not in st.session_state: 
    st.session_state.temperature = 0.0
if "show_refs" not in st.session_state: 
    st.session_state.show_refs = True  # Always true now, toggle is hidden
if "references_last" not in st.session_state:
    st.session_state.references_last = ""
if "answer_last" not in st.session_state:
    st.session_state.answer_last = ""
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "last_processed_files" not in st.session_state:
    st.session_state.last_processed_files = []
# Hybrid RAG settings - linked to config
if "use_hybrid_rag" not in st.session_state:
    st.session_state.use_hybrid_rag = True
if "enable_multimodal" not in st.session_state:
    st.session_state.enable_multimodal = True
if "enable_graph" not in st.session_state:
    st.session_state.enable_graph = True
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ---------- BRAND CSS ----------
# Skip corporate UI loading for faster startup
base_css = ""

# Stage-7 Theme: Violet (#342D8C), Opal (#1CCFC9), Navy (#131838)
stage7_css = base_css + """
/* --- Theme Colors --- */
:root {
  --violet: #342D8C;
  --opal: #1CCFC9;
  --navy: #131838;
}

/* Global typography and compact spacing */
html, body, [data-testid="stAppViewContainer"], .main {background: #ffffff;}
.main {padding-top: 0.5rem !important;}
.element-container {margin-bottom: 0.4rem !important;}
[data-testid="stSidebar"] {padding: 0.75rem 0.75rem 0.5rem !important;}
[data-testid="stSidebar"] h2 {font-size: 1.05rem; margin: 0.25rem 0 0.5rem 0; color: var(--navy);}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {font-size: 0.95rem; margin: 0.25rem 0 0.5rem 0; color: var(--navy);}
[data-testid="stSidebar"] * {text-align: left !important;}
[data-testid="stFileUploader"] {padding: 0.25rem 0 !important;}
.stSlider > div {padding-bottom: 0.25rem !important;}

/* Header bars */
.app-header {background: var(--violet); padding: 0.6rem 0.8rem; border-radius: 10px;}
.app-header h1 {color: #fff; margin: 0; font-weight: 700; font-size: 1.4rem;}
.login-header {background: var(--violet); padding: 0.6rem 0.8rem; border-radius: 10px;}
.login-header h1 {color: #ffffff; margin: 0; font-weight: 700; font-size: 1.4rem;}

/* Cards */
.answer-card {padding: 0.9rem; border-radius: 10px; border: 1px solid #E6F7F6; background: #FFFFFF;}
.sources-card {padding: 0.9rem; border-radius: 10px; border: 1px solid #CBEFED; background: #F7FFFE;}

/* Citation links */
.citation-link {
  color: #1f77b4 !important;
  text-decoration: none !important;
  font-weight: 500;
  font-size: 0.75em !important;
  cursor: pointer !important;
  border-bottom: 1px dotted #1f77b4;
  transition: all 0.2s ease;
  padding: 0px 1px;
  border-radius: 2px;
  background-color: rgba(31, 119, 180, 0.08);
  display: inline-block;
  margin: 0 0px;
  user-select: none;
}
.citation-link:hover {
  color: #0d5aa7 !important;
  border-bottom: 1px solid #0d5aa7 !important;
  text-decoration: none !important;
  background-color: rgba(31, 119, 180, 0.15) !important;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(31, 119, 180, 0.2);
}
.citation-link:active {
  transform: translateY(0);
  background-color: rgba(31, 119, 180, 0.25) !important;
}

/* Source document blocks - compact spacing */
.source-block {
  background: #ffffff;
  border: 1px solid #e6f7f6;
  border-radius: 8px;
  padding: 0.7rem;
  margin-bottom: 0.5rem;
  transition: all 0.2s ease;
}
.source-block:hover {
  border-color: var(--opal);
  box-shadow: 0 2px 8px rgba(28, 207, 201, 0.1);
}
.source-filename {
  color: var(--navy);
  font-weight: 600;
  font-size: 1.02em;
  margin-bottom: 0.2rem;
}
.source-page {
  color: var(--opal);
  font-weight: 400;
}
.source-section {
  color: #7a8199;
  font-size: 0.9em;
  margin-bottom: 0.3rem;
}
.source-snippet {
  background: #f8fffe;
  border-left: 3px solid var(--opal);
  padding: 0.5rem 0.7rem;
  margin: 0.3rem 0;
  border-radius: 6px;
  color: var(--navy);
  font-size: 0.92em;
  line-height: 1.4;
}
.source-divider {
  border: none;
  border-top: 1px solid #e6e9f2;
  margin: 0.5rem 0;
}

/* Buttons: Opal base with Violet hover (cover normal + form submit variants) */
.stButton button,
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="baseButton-primaryFormSubmit"],
.stForm button,
.stForm button[type="submit"] {
  background: var(--opal) !important;
  color: var(--navy) !important;
  border: 1px solid var(--opal) !important;
  box-shadow: none !important;
}
.stButton button:hover,
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-primaryFormSubmit"]:hover,
.stForm button:hover,
.stForm button[type="submit"]:hover {
  background: #18bab4 !important; /* slightly darker opal */
  border-color: var(--violet) !important;
}
/* Secondary buttons */
button[kind="secondary"], .stButton button[kind="secondary"] {
  background: #f5f7fb !important; color: var(--navy) !important; border: 1px solid #e6e9f2 !important;
}
button[kind="secondary"]:hover {border-color: var(--opal) !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {height: 34px; padding: 0 12px; background: #f2f3f8; border-radius: 8px;}
.stTabs [data-baseweb="tab"]:hover {background: #e9eaf3;}
.stTabs [aria-selected="true"] {background: var(--violet) !important; color: #fff;}

/* Inputs */
input, textarea {border-radius: 8px !important;}
input[type="checkbox"], input[type="radio"], input[type="range"] {accent-color: var(--opal) !important;}
/* Ensure input text colors remain neutral/navy (not Opal) */
input[type="text"], input[type="password"], input[type="email"], textarea {color: var(--navy) !important; caret-color: var(--navy) !important;}
input::placeholder, textarea::placeholder {color: #7a8199 !important;}
/* Edge-specific reveal/clear icons - keep default coloring */
input::-ms-reveal, input::-ms-clear {filter: none !important; color: inherit !important;}
/* Slider track/thumb (best-effort across engines) */
input[type="range"]::-webkit-slider-thumb {background: var(--opal) !important; border: 1px solid var(--opal) !important;}
input[type="range"]::-webkit-slider-runnable-track {background: linear-gradient(90deg, var(--opal), #d8f7f5) !important;}
input[type="range"]::-moz-range-thumb {background: var(--opal) !important; border: 1px solid var(--opal) !important;}
input[type="range"]::-moz-range-track {background: #e8f8f7 !important;}
/* Switch/toggle when checked */
[data-testid="stSwitch"] div[role="switch"][aria-checked="true"] {background: var(--opal) !important; border-color: var(--opal) !important;}
/* Selectbox focus and selected option */
.stSelectbox [data-baseweb="select"] div:focus {box-shadow: 0 0 0 1px var(--opal) !important;}
.stSelectbox [data-baseweb="select"] div[aria-selected="true"] {background: #d8f7f5 !important;}

/* Reduce vertical whitespace in chat */
.stChatMessage {margin-bottom: 0.4rem !important;}

/* Expander */
div[data-testid="stExpander"] {border: 1px solid #e6e9f2; border-radius: 8px;}

/* Tabs styling - now at bottom, no need for sticky */
.stTabs {
  background: #ffffff !important;
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
  margin-top: 0.5rem !important;
  margin-bottom: 0.5rem !important;
  border-top: 1px solid #e6e9f2;
}

/* Chat avatar icon colors (best-effort selectors) */
.stChatMessage [data-testid="stChatMessageAvatar"] svg {color: var(--violet) !important; fill: var(--violet) !important;}
.stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageAvatar"] svg {color: var(--opal) !important; fill: var(--opal) !important;}
/* Additional heuristic: if role label exists */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) [data-testid="stChatMessageAvatar"] svg {color: var(--violet) !important;}
[data-testid="stChatMessage"]:has(.user) [data-testid="stChatMessageAvatar"] svg {color: var(--opal) !important;}

/* Password input reveal toggle uses neutral grey, not Opal */
/* Prefer explicit selectors for BaseWeb input eye button */
/* ===== PASSWORD EYE TOGGLE (transparent, match input) ===== */
[data-baseweb="input"] button,
input[type="password"] + div button {
  background: transparent !important;    /* match input background */
  border: 1px solid transparent !important; /* no visible border */
  color: var(--navy) !important;         /* same as input text */
}
[data-baseweb="input"] button:hover,
input[type="password"] + div button:hover {
  background: transparent !important;    /* keep background consistent */
  border-color: transparent !important;  /* no border on hover */
  color: var(--navy) !important;         /* keep icon color */
}
/* Ensure the eye SVG inherits Navy (same as input text) */
[data-baseweb="input"] button svg,
input[type="password"] + div button svg {
  fill: var(--navy) !important;
  stroke: var(--navy) !important;
}
"""

st.markdown(f"<style>{stage7_css}</style>", unsafe_allow_html=True)

# Sidebar toggle and slider theming (Opal accents, Navy labels)
SIDEBAR_TOGGLE_SLIDER_CSS = """
<style>
/* ===== Toggle (Show References) ===== */
[data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
  background-color: #1CCFC9 !important;  /* Opal */
  border-color: #1CCFC9 !important;
}
[data-testid="stSidebar"] [role="switch"][aria-checked="false"] {
  background-color: #dcdcdc !important;  /* grey off state */
  border-color: #dcdcdc !important;
}

/* ===== Slider (Answer Style) ===== */
.stSlider [role="slider"] {
  background-color: #1CCFC9 !important;  /* Opal thumb */
}
.stSlider .st-ce,   /* newer streamlit builds thumb/track */
.stSlider .st-bf,
.stSlider .st-af,
.stSlider .css-1ld3v1b {
  background-color: #1CCFC9 !important;  /* Opal track fill */
}
.stSlider label, .stSlider div {
  color: #131838 !important;             /* Navy for text/labels */
}
</style>
"""
st.markdown(SIDEBAR_TOGGLE_SLIDER_CSS, unsafe_allow_html=True)

# --- Themed avatar icons as data URIs ---
def _svg_circle_data_uri(color_hex: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="{color_hex}"/></svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

AVATAR_USER = _svg_circle_data_uri("#1CCFC9")  # Opal
AVATAR_ASSISTANT = _svg_circle_data_uri("#342D8C")  # Violet

# ---------- CORE IMPORTS ----------
import json
# Delay heavy imports until needed for faster startup

# ---------- LOGIN ----------
def login_view():
    """Display login form."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class='login-header' style='text-align:center;'>
            <h1>Knowledge Management Assistant</h1>
        </div>
        <p style='color:#131838; margin: 0.5rem 0 0.75rem 0; text-align:center;'>Secure login</p>
        """, unsafe_allow_html=True)
        
        with st.form("login-form", clear_on_submit=True):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submitted:
                if username == USERNAME and password == PASSWORD:
                    st.session_state.auth = True
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

# ---------- SOURCE RENDERING HELPER ----------
def make_citations_clickable(text):
    """Convert inline citations to clickable links that jump to Sources section."""
    import html
    import re

    escaped_text = html.escape(text)

    # Pattern to match [1], [2], [1][2], [1-3], etc.
    citation_pattern = r'\[(\d+(?:-\d+)?(?:\]\[\d+(?:-\d+)?)*)\]'

    def replace_citation(match):
        citation_text = match.group(0)  # e.g., "[1]", "[1][2]", "[1-3]"
        # Extract all numbers from the citation
        numbers = re.findall(r'\d+', citation_text)
        # Create a clickable link that will switch to Sources tab and scroll
        if numbers:
            first_number = numbers[0]
            # Return a styled span that looks clickable and triggers tab switch + scroll
            return f'<a href="#source-{first_number}" class="citation-link" data-source="{first_number}" style="color: #1CCFC9; text-decoration: none; cursor: pointer;" title="View source {first_number}">{citation_text}</a>'
        return citation_text

    return re.sub(citation_pattern, replace_citation, escaped_text)

def render_sources_detailed(references_json: str):
    """Render source documents in a structured format with rich details."""
    try:
        # Parse JSON string from ChatBot.clean_references
        references = json.loads(references_json) if references_json else []

        if not references:
            return st.info("No source documents found for this query.")

        # Add custom CSS for source blocks with Opal + Navy theme
        st.markdown("""
        <style>
        .source-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
            border: 2px solid #e6f9f7;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(28, 207, 201, 0.1);
        }
        .source-card:hover {
            box-shadow: 0 4px 12px rgba(28, 207, 201, 0.2);
            border-color: #1CCFC9;
            transform: translateY(-2px);
        }
        .source-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid #e6f9f7;
            padding-bottom: 8px;
        }
        .source-filename {
            color: #131838;
            font-weight: 700;
            font-size: 17px;
            flex-grow: 1;
            letter-spacing: -0.02em;
        }
        .source-page {
            background: linear-gradient(135deg, #1CCFC9 0%, #17b0aa 100%);
            color: white;
            font-weight: 600;
            font-size: 13px;
            padding: 4px 10px;
            border-radius: 20px;
            margin-left: 12px;
        }
        .source-section {
            color: #7a8199;
            font-size: 13px;
            font-style: italic;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        .source-snippet {
            background: #f7fffe;
            color: #4a5168;
            font-size: 14px;
            line-height: 1.6;
            border-left: 4px solid #1CCFC9;
            padding: 12px 16px;
            margin-top: 12px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        }
        .source-detail-number {
            display: inline-block;
            color: #9ca3af;  /* Muted gray */
            font-weight: 500;
            font-size: 13px;
            margin-right: 8px;
        }
        .source-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #e6f9f7, transparent);
            margin: 24px 0;
        }
        /* Gray styling for View buttons */
        .stLinkButton > a {
            background-color: #f3f4f6 !important;
            color: #6b7280 !important;
            border: 1px solid #d1d5db !important;
        }
        .stLinkButton > a:hover {
            background-color: #e5e7eb !important;
            color: #4b5563 !important;
            border-color: #9ca3af !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Render each source as a structured block
        for i, ref in enumerate(references, 1):
            # Create columns for source display and action button
            col1, col2 = st.columns([5, 1])

            with col1:
                # Create source block without bullet points with anchor link
                source_number = ref.get("number", i)
                source_html = f'<div class="source-card" id="source-{source_number}">'

                # Header with source number, filename and page
                source_html += '<div class="source-header">'
                source_html += f'<span class="source-detail-number">[{source_number}]</span>'
                source_html += f'<span class="source-filename">📄 {ref["filename"]}</span>'
                if ref.get('page'):
                    source_html += f'<span class="source-page">Page {ref["page"]}</span>'
                source_html += '</div>'

                # Section heading if available
                if ref.get('section') and ref['section']:
                    source_html += f'<div class="source-section">{ref["section"]}</div>'

                # Snippet with proper formatting
                snippet = ref.get('snippet', '')
                if snippet:
                    # Escape HTML in snippet to prevent injection
                    import html
                    escaped_snippet = html.escape(snippet)
                    source_html += f'<div class="source-snippet">{escaped_snippet}</div>'

                source_html += '</div>'

                # Render the complete source block
                st.markdown(source_html, unsafe_allow_html=True)

            with col2:
                # Center the button vertically
                st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)

                # Check for URL in metadata
                if ref.get('url'):
                    # Add viewer parameter to force browser viewing instead of download
                    viewer_url = f"{ref['url']}#view=FitH"
                    # Render clickable link button for Azure Blob Storage URL that opens in new tab
                    st.markdown(
                        f"""
                        <a href="{viewer_url}" target="_blank" rel="noopener noreferrer" style="
                            display: inline-block;
                            width: 100%;
                            padding: 0.5rem 1rem;
                            background-color: #f5f7fb;
                            color: #0f2344;
                            text-align: center;
                            text-decoration: none;
                            border: 1px solid #e6e9f2;
                            border-radius: 0.375rem;
                            font-weight: 500;
                            transition: all 0.2s;
                        " onmouseover="this.style.backgroundColor='#e6e9f2'; this.style.borderColor='#18bab4';"
                           onmouseout="this.style.backgroundColor='#f5f7fb'; this.style.borderColor='#e6e9f2';">
                            View
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # No viewing option available for local files in New document mode
                    st.button(
                        "Local",
                        disabled=True,
                        use_container_width=True,
                        help="Local document - no external link available",
                        key=f"local_{i}_{ref.get('filename', 'doc')[:20]}"
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            # Add divider between sources (except after last one)
            if i < len(references):
                st.markdown('<hr class="source-divider">', unsafe_allow_html=True)

    except json.JSONDecodeError:
        # Fallback for old markdown format
        st.warning("Source format needs updating. Showing raw references:")
        st.markdown(references_json)
    except Exception as e:
        st.error(f"Error rendering sources: {str(e)}")


def extract_cited_numbers_from_answer(answer_text):
    """Extract citation numbers [1], [2], etc. from the answer text."""
    import re
    # Find all citations in the format [1], [2], [1][2], etc.
    citation_pattern = r'\[(\d+)\]'
    cited_numbers = set()

    for match in re.finditer(citation_pattern, answer_text):
        cited_numbers.add(int(match.group(1)))

    return sorted(list(cited_numbers))

def render_sources(references_json: str):
    """Render source documents in a simple list format - only showing cited sources."""
    try:
        # Parse JSON string from ChatBot.clean_references
        references = json.loads(references_json) if references_json else []

        if not references:
            return st.info("No source documents found for this query.")

        # Extract citation numbers from the answer
        answer_text = st.session_state.get("answer_last", "")
        cited_numbers = extract_cited_numbers_from_answer(answer_text)

        if not cited_numbers:
            return  # No citations found, don't show sources

        # Filter references to only show cited ones
        cited_references = [ref for ref in references if ref.get("number", 0) in cited_numbers]

        if not cited_references:
            return

        # Add minimal CSS for simple source list display with smaller font
        st.markdown("""
        <style>
        .sources-list {
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
            padding: 12px;
            background: #f9fafb;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: auto;
        }
        .source-item {
            margin-bottom: 6px;
            color: #374151;
            font-size: 12px;
        }
        .source-number {
            color: #6b7280;
            font-weight: bold;
        }
        .source-view {
            color: #1CCFC9;
            text-decoration: none;
            margin-left: 5px;
            font-weight: 500;
        }
        .source-view:hover {
            text-decoration: underline;
            color: #17b0aa;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display "Sources:" header
        st.markdown("**Sources:**")

        # Build the sources list HTML
        sources_html = '<div class="sources-list">'

        for ref in cited_references:
            source_number = ref.get("number", 0)
            filename = ref.get("filename", "Unknown")
            page = ref.get("page", "N/A")
            url = ref.get("url", None)

            # Format: [1] filename.pdf - Page X View
            source_line = f'<div class="source-item" id="source-{source_number}">'
            source_line += f'<span class="source-number">[{source_number}]</span> '
            source_line += f'{filename} - Page {page}'

            if url:
                # Add viewer parameter to force browser viewing instead of download
                viewer_url = f"{url}#view=FitH"
                source_line += f' <a href="{viewer_url}" target="_blank" rel="noopener noreferrer" class="source-view">View</a>'

            source_line += '</div>'
            sources_html += source_line

        sources_html += '</div>'

        # Render the complete sources list
        st.markdown(sources_html, unsafe_allow_html=True)

    except json.JSONDecodeError:
        # Fallback for old markdown format
        st.warning("Source format needs updating. Showing raw references:")
        st.markdown(references_json)
    except Exception as e:
        st.error(f"Error rendering sources: {str(e)}")

# ---------- SYNC CONFIG HELPER ----------
def sync_config_with_session():
    """Sync session state toggles with config file."""
    try:
        from utils.load_config import LoadConfig
        cfg = LoadConfig()
        
        # Update config based on session state
        cfg.hybrid_rag['enabled'] = st.session_state.use_hybrid_rag
        cfg.hybrid_rag['multimodal']['enabled'] = st.session_state.enable_multimodal
        cfg.hybrid_rag['graph']['enabled'] = st.session_state.enable_graph
        cfg.hybrid_rag['router']['debug'] = st.session_state.debug_mode
        
        # Temperature is already passed directly to ChatBot.respond()
        # so we don't need to modify config for that
        
        return True
    except Exception as e:
        print(f"Config sync warning: {e}")
        return False

# ---------- MAIN APP ----------
def app_view():
    """Display main application after authentication."""
    # Import heavy modules only when app is loaded (not during login)
    from utils.chatbot import ChatBot

    # Header (Violet bar)
    st.markdown("""
    <div class="app-header">
        <h1>Knowledge Management Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls - compact layout
    with st.sidebar:
        # Sidebar header
        st.markdown("## Settings")
        st.session_state.rag_option = st.selectbox(
            "Ask Mode",
            ["Current documents", "New document"],
            index=0 if st.session_state.rag_option == "Current documents" else 1,
            help="Choose where answers come from"
        )

        # Always show references - no toggle needed
        st.session_state.show_refs = True

        # Temperature control
        st.session_state.temperature = st.slider(
            "Answer Style",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.1,
            help="0 = Factual & precise | 1 = More creative & exploratory"
        )

        # Upload documents
        st.markdown("#### Upload documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "doc", "docx", "txt", "csv", "html", "eml", "msg", "md"],
            accept_multiple_files=True,
            help="Add documents to the knowledge base",
            label_visibility="collapsed"
        )
        
        # Detect file changes (additions/removals)
        current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
        files_changed = current_file_names != st.session_state.uploaded_file_names
        
        # Only handle clearing when ALL files are removed in New document mode
        if st.session_state.rag_option == "New document" and files_changed:
            if len(current_file_names) == 0 and len(st.session_state.uploaded_file_names) > 0:
                # All files were removed - clear the vector database completely
                st.info("Clearing New document database...")
                import shutil
                from utils.load_config import LoadConfig
                cfg = LoadConfig()
                
                # Clear the new uploads directory
                if NEW_DOC_DIR.exists():
                    try:
                        shutil.rmtree(NEW_DOC_DIR)
                        NEW_DOC_DIR.mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        # If can't remove directory, at least remove files
                        for item in os.listdir(NEW_DOC_DIR):
                            try:
                                (NEW_DOC_DIR / item).unlink()
                            except:
                                pass
                
                # Try to clear vector database
                if os.path.exists(cfg.custom_persist_directory):
                    try:
                        shutil.rmtree(cfg.custom_persist_directory)
                    except PermissionError:
                        # Can't fully remove, but don't create marker - just let it rebuild naturally
                        st.warning("Database files are locked. Will be replaced on next upload.")
                
                st.session_state.last_processed_files = []
                st.success("Ready for new documents")
            # Removed the problematic elif block that treated file changes as needing rebuild
        
        # Update session state
        st.session_state.uploaded_file_names = current_file_names
        
        if uploaded_files:
            saved_files = []
            # Save to appropriate directory based on mode
            if st.session_state.rag_option == "New document":
                save_dir = NEW_DOC_DIR
                # Clear the directory for New document mode - each upload replaces previous
                if save_dir.exists():
                    try:
                        for item in os.listdir(save_dir):
                            item_path = save_dir / item
                            try:
                                if item_path.is_file():
                                    item_path.unlink()
                            except PermissionError:
                                pass  # Skip locked files
                    except Exception:
                        pass
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = DATA_DIR / "docs"
            
            for uploaded_file in uploaded_files:
                output_path = save_dir / uploaded_file.name
                with open(output_path, "wb") as f:
                    f.write(uploaded_file.read())
                saved_files.append(uploaded_file.name)
            
            st.success(f"Saved {len(saved_files)} file(s)")
            
            # Auto-process if in New document mode
            if st.session_state.rag_option == "New document" and len(saved_files) > 0:
                with st.spinner("Auto-processing documents..."):
                    # Process with clearing (but handling permission errors)
                    status_msg = run_manual_upload(mode="new", clear_existing=True)
                    if "successfully" in status_msg or "updated" in status_msg.lower():
                        st.success("Documents processed!")
                        st.session_state.last_processed_files = saved_files
                    elif "up-to-date" in status_msg:
                        st.info("Documents already processed")
                    elif "No new documents" in status_msg or "No documents to process" in status_msg:
                        st.info("No documents to process")
                    else:
                        # Don't show error for empty directory
                        if "Directory does not exist" not in status_msg:
                            st.warning(f"Processing status: {status_msg}")
            else:
                st.info("Click 'Run Data Upload' or switch to 'New document' mode")
        elif st.session_state.rag_option == "New document":
            # No files uploaded
            if st.session_state.last_processed_files:
                # We had processed files before - DB should be cleared
                st.session_state.last_processed_files = []
            st.info("Upload documents to use 'New document' mode")
        
        # Run Data Upload button (grey secondary with Opal hover)
        if st.button("Run Data Upload", use_container_width=True, type="secondary"):
            with st.spinner("Processing documents..."):
                # Process based on current RAG mode
                if st.session_state.rag_option == "New document":
                    status_msg = run_manual_upload(mode="new")
                else:
                    status_msg = run_manual_upload(mode="current")
                st.session_state.upload_status = status_msg
                if "successfully" in status_msg or "updated" in status_msg.lower():
                    st.success(status_msg)
                elif "warning" in status_msg.lower():
                    st.warning(status_msg)
                elif "error" in status_msg.lower():
                    st.error(status_msg)
                else:
                    st.info(status_msg)
        
        # Advanced Features (collapsible)
        with st.expander("Advanced Features", expanded=False):
            st.session_state.use_hybrid_rag = st.checkbox(
                "Hybrid Router",
                value=st.session_state.use_hybrid_rag,
                help="Enable intelligent routing between retrieval strategies"
            )
            st.session_state.enable_multimodal = st.checkbox(
                "Multimodal Search",
                value=st.session_state.enable_multimodal,
                disabled=not st.session_state.use_hybrid_rag,
                help="Enable image + text retrieval (requires Hybrid Router)"
            )
            st.session_state.enable_graph = st.checkbox(
                "Knowledge Graph",
                value=st.session_state.enable_graph,
                disabled=not st.session_state.use_hybrid_rag,
                help="Enable graph-based retrieval (requires Hybrid Router)"
            )
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.debug_mode,
                help="Show detailed processing information"
            )
            # Sync config when toggles change
            sync_config_with_session()

        # Quick Guide (collapsed by default)
        with st.expander("Quick Guide", expanded=False):
            st.markdown("""
            - **Ask Mode**: Choose 'Current documents' or 'New document'.
            - **Upload documents**: Add files (PDF, DOCX, TXT, etc.).
            - **Run Data Upload**: Build/update the index for the selected mode.
            - **Ask**: Type your question below; answers cite sources when enabled.
            """, unsafe_allow_html=True)
        
        # Logout button at the bottom (no extra divider to save space)
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.auth = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat area - display all messages first
    chat_container = st.container()
    with chat_container:
        # Display full chat history
        for message in st.session_state.chat_history:
            _avatar = AVATAR_ASSISTANT if message["role"] == "assistant" else AVATAR_USER
            with st.chat_message(message["role"], avatar=_avatar):
                if message["role"] == "assistant":
                    # Display answer in a styled card with clickable citations
                    clickable_content = make_citations_clickable(message["content"])
                    st.markdown(f"""
                    <div class="answer-card">
                        {clickable_content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
    
    # Tabs at the bottom - just above input field
    # These tabs show the latest response details
    if st.session_state.chat_history:
        st.markdown("---")  # Separator line

        # Only show tabs in "Current documents" mode
        if st.session_state.rag_option == "Current documents":
            # Create tabs - Sources tab will be shown by default
            tabs = st.tabs(["Latest Answer", "Sources"])

            with tabs[0]:
                # Show the last assistant response
                last_assistant_msg = None
                for msg in reversed(st.session_state.chat_history):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg
                        break

                if last_assistant_msg:
                    clickable_content = make_citations_clickable(last_assistant_msg["content"])
                    st.markdown(f"""
                    <div class="answer-card">
                        {clickable_content}
                    </div>
                    """, unsafe_allow_html=True)

                    # Add sources list directly below the answer
                    if st.session_state.references_last:
                        render_sources(st.session_state.references_last)
                else:
                    st.info("No answer available yet.")

            # Sources tab - only for Current documents mode
            with tabs[1]:
                # Display sources/references in structured format
                if st.session_state.references_last:
                    # Add JavaScript for citation click handling
                    st.markdown("""
                    <script>
                    // Function to handle citation clicks
                    window.handleCitationClick = function(sourceNumber) {
                        // Find and scroll to the source
                        const sourceElement = document.querySelector('#source-' + sourceNumber);
                        if (sourceElement) {
                            // Scroll the source into view
                            sourceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });

                            // Highlight the source temporarily
                            sourceElement.style.transition = 'all 0.3s ease';
                            sourceElement.style.backgroundColor = '#e6f9f7';
                            sourceElement.style.border = '2px solid #1CCFC9';
                            sourceElement.style.padding = '18px';

                            setTimeout(() => {
                                sourceElement.style.backgroundColor = '';
                                sourceElement.style.border = '';
                                sourceElement.style.padding = '';
                            }, 2500);
                        }
                    }

                    // Add click listeners to all citation links when page loads
                    document.addEventListener('DOMContentLoaded', function() {
                        document.addEventListener('click', function(e) {
                            if (e.target && e.target.classList && e.target.classList.contains('citation-link')) {
                                e.preventDefault();
                                const sourceNum = e.target.getAttribute('data-source');
                                if (sourceNum) {
                                    // First, try to switch to Sources tab programmatically
                                    // Find and click the Sources tab button
                                    const tabButtons = document.querySelectorAll('[data-testid="stTab"]');
                                    if (tabButtons && tabButtons.length > 1) {
                                        tabButtons[1].click(); // Click Sources tab (index 1)
                                        setTimeout(() => {
                                            handleCitationClick(sourceNum);
                                        }, 100);
                                    } else {
                                        // If tabs not found, just try to scroll
                                        handleCitationClick(sourceNum);
                                    }
                                }
                            }
                        });
                    });
                    </script>
                    """, unsafe_allow_html=True)

                    # Render sources in a scrollable container with detailed view
                    st.markdown('<div class="sources-container" style="max-height: 600px; overflow-y: auto; padding: 1rem;">', unsafe_allow_html=True)
                    render_sources_detailed(st.session_state.references_last)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No sources available. Sources will appear here after you ask a question.")
        else:
            # For "New document" mode, just show the latest answer without tabs
            st.markdown("#### Latest Answer")
            last_assistant_msg = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg
                    break

            if last_assistant_msg:
                clickable_content = make_citations_clickable(last_assistant_msg["content"])
                st.markdown(f"""
                <div class="answer-card">
                    {clickable_content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No answer available yet.")
    
    # Chat input (outside tabs, always visible)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Switch to Answer tab automatically
        # Note: Streamlit doesn't provide direct tab switching, but the message will appear in tab1
        
        # Display user message in main chat area
        with chat_container:
            with st.chat_message("user", avatar=AVATAR_USER):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
                with st.spinner("Thinking..."):
                    try:
                        # Sync config before querying
                        sync_config_with_session()
                        
                        # Call ChatBot.respond with current settings
                        answer_text, new_history, refs_md = ChatBot.respond(
                            st.session_state.chat_history[:-1],  # Exclude the just-added message
                            prompt,
                            st.session_state.rag_option,
                            st.session_state.temperature
                        )
                        
                        # Display answer with clickable citations
                        clickable_answer = make_citations_clickable(answer_text)
                        st.markdown(f"""
                        <div class="answer-card">
                            {clickable_answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer_text
                        })
                        
                        # Store references and answer for the Sources tab
                        if refs_md:
                            st.session_state.references_last = refs_md
                            st.session_state.answer_last = answer_text  # Store answer for citation filtering
                            # Show a subtle indicator that sources are available
                            st.caption("View source documents in the 'Sources' tab below")
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Force a rerun to update the UI
        st.rerun()

# ---------- UPLOAD ROUTINE ADAPTER ----------
def run_manual_upload(mode="current", clear_existing=False):
    """
    Calls the existing upload/ingest function and returns a status string.
    Captures stdout to parse status messages.
    
    Args:
        mode: "current" for updating main vector DB, "new" for new document mode
        clear_existing: If True, clear existing vector DB before processing (for New document mode)
    """
    try:
        import io
        from contextlib import redirect_stdout
        from upload_data_manually import upload_data_for_mode
        
        # Capture output from upload_data_manually
        f = io.StringIO()
        with redirect_stdout(f):
            upload_data_for_mode(mode=mode, clear_existing=clear_existing)
        
        output = f.getvalue()
        
        # Parse output for user-friendly messages
        if "WARNING: Cannot load" in output and "unstructured" in output:
            # Extract file types that need unstructured
            import re
            failed_types = set()
            if ".doc" in output or ".docx" in output:
                failed_types.add(".doc/.docx")
            if ".eml" in output or ".msg" in output:
                failed_types.add(".eml/.msg") 
            if ".html" in output or ".htm" in output:
                failed_types.add(".html")
            
            type_list = ", ".join(failed_types) if failed_types else "some file types"
            return (f"Cannot process {type_list} files - missing 'unstructured' package.\n"
                    f"To enable support, install: pip install unstructured python-docx\n"
                    f"Failed files are NOT marked as processed and can be retried.")
        elif "WARNING: No text could be extracted" in output:
            if "Missing dependencies" in output:
                return "No text extracted. This may be due to missing dependencies or unsupported formats. Failed files can be retried."
            else:
                return "Warning: Some files have no extractable text (may be scanned PDFs). Skipped embedding."
        elif "No new documents to process" in output:
            return "No new documents found. Vector database is up-to-date."
        elif "Updated VectorDB with" in output:
            # Extract the number of documents if possible
            import re
            match = re.search(r"Updated VectorDB with (\d+)", output)
            failed_match = re.search(r"Failed to load (\d+) file", output)
            
            result_msg = ""
            if match:
                num_docs = match.group(1)
                result_msg = f"Successfully processed and indexed {num_docs} document(s)."
            else:
                result_msg = "Documents processed and vector database updated."
            
            if failed_match:
                num_failed = failed_match.group(1)
                result_msg += f"\n{num_failed} file(s) failed to load (check console for details)."
            
            return result_msg
        elif "processed successfully" in output:
            return "Documents processed successfully and added to vector database."
        elif "Error" in output:
            return f"Error during processing: {output[:200]}"
        else:
            return "Upload process completed. Check logs for details."
            
    except ImportError as e:
        return f"Import error: {str(e)}. Check that all dependencies are installed."
    except Exception as e:
        return f"Error during data upload: {str(e)}"

# ---------- MAIN ROUTER ----------
def main():
    """Main application entry point."""
    # Production auth flow
    if not st.session_state.auth:
        login_view()
    else:
        app_view()

if __name__ == "__main__":
    main()