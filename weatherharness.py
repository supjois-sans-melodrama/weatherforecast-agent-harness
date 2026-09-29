import json
import os
import requests
import numpy as np
from datetime import datetime, date, timedelta
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# --- PAGE CONFIGURATION & CUSTOM STYLING ---
st.set_page_config(
    page_title="Global Weather Intelligence | Agent Harness + ChromaDB & API",
    page_icon="🌤️",
    layout="wide"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Reddit+Sans:ital,wght@0,300..900;1,300..900&family=Belleza&display=swap" rel="stylesheet">

<style>
    html, body, [class*="css"], div, span, p, label {
        font-family: 'Reddit Sans', 'Belleza', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    code, pre, .stCodeBlock {
        font-family: 'Fira Code', monospace !important;
    }
    .stApp { 
        background-color: #0b0f17; 
        color: #e2e8f0; 
    }
    
    .main-header {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
        line-height: 1.2 !important;
    }

    .disclaimer-banner {
        background-color: #1e1b4b;
        border: 1px solid #4338ca;
        color: #c7d2fe;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .form-step-label {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .form-step-number {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff;
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    div[data-testid="stRadio"] label {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        color: #f1f5f9 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 14px !important;
    }
    .stMarkdown table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #334155 !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
    }
    .stMarkdown th {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%) !important;
        color: #c7d2fe !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.08em !important;
        padding: 14px 16px !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
    .stMarkdown td {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #1f2937 !important;
        font-size: 1rem !important;
    }
    .stMarkdown tr:nth-child(even) td {
        background-color: #1f2937 !important;
    }
    .stMarkdown tr:hover td {
        background-color: #374151 !important;
        color: #60a5fa !important;
        transition: all 0.2s ease-in-out !important;
    }
    .harness-card {
        background-color: #131b2e;
        border-left: 4px solid #818cf8;
        padding: 14px 18px;
        margin-bottom: 14px;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .harness-badge {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78em;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #60a5fa !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid #1e293b !important;
        padding-bottom: 4px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. AUTOMATIC API KEY INITIALIZATION ---
api_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)


# --- 2. CHROMADB VECTOR RAG (DYNAMIC SCAN OF `./knowledge_base` DIR) ---
CHROMA_DB_DIR = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

def load_txt_files_from_dir(dir_path: str):
    """Scans `./knowledge_base` for all .txt files and extracts their content."""
    docs = []
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return docs

    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        doc_id = os.path.splitext(filename)[0]
                        docs.append({"id": doc_id, "text": text, "source": filename})
            except Exception as e:
                st.warning(f"Failed to read file {filename}: {e}")
    return docs

@st.cache_resource
def get_chroma_collection():
    """Initializes ChromaDB and ingests .txt files directly from `./knowledge_base` folder."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        name="weather_advisories",
        embedding_function=ef
    )
    
    # Load all files from knowledge_base folder
    file_docs = load_txt_files_from_dir(KNOWLEDGE_BASE_DIR)
    
    if file_docs:
        existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
        
        new_docs = [doc["text"] for doc in file_docs if doc["id"] not in existing_ids]
        new_ids = [doc["id"] for doc in file_docs if doc["id"] not in existing_ids]
        new_meta = [{"source": doc["source"]} for doc in file_docs if doc["id"] not in existing_ids]
        
        if new_ids:
            collection.add(documents=new_docs, ids=new_ids, metadatas=new_meta)
            
    return collection

chroma_collection = get_chroma_collection()

def search_knowledge_base(query: str) -> str:
    """RAG Tool: Performs vector similarity search using persistent ChromaDB."""
    try:
        results = chroma_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        docs = results.get("documents", [[]])[0]
        if not docs:
            return json.dumps({"results": ["No relevant advisories or policy documents found in knowledge base."]})
            
        return json.dumps({"retrieved_context": docs})
    except Exception as e:
        return json.dumps({"error": f"Failed to execute ChromaDB RAG search: {str(e)}"})


# --- 3. DEDUPLICATED AUTO-SUGGESTION HELPER ---
@st.cache_data(ttl=3600)
def search_cities(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=10"
        res = requests.get(geo_url, timeout=5).json()
        results = res.get("results", [])
        suggestions = []
        for item in results:
            name = item.get("name")
            country = item.get("country", "")
            admin = item.get("admin1", "")
            label = f"{name}, {country}" if not admin else f"{name}, {admin}, {country}"
            suggestions.append(label)
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:5]
    except Exception:
        return []


# --- 4. WEATHER TOOL IMPLEMENTATION ---
def get_weather(location: str, forecast_type: str = "current", start_date: str = None, temperature_unit: str = "celsius") -> str:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return json.dumps({"error": f"Location '{location}' not found. Please provide a valid city name."})

        result = geo_res["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        city_name, country = result["name"], result.get("country", "")
        
        unit_param = "fahrenheit" if temperature_unit.lower() == "fahrenheit" else "celsius"
        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&timezone=auto&temperature_unit={unit_param}"

        if forecast_type == "hourly":
            date_param = f"&start_date={start_date}&end_date={start_date}" if start_date else "&forecast_days=1"
            url = f"{base_url}&hourly=temperature_2m,precipitation_probability{date_param}"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "temperature_unit": unit_param.capitalize(),
                "timescale": f"Hourly forecast for {start_date if start_date else 'today'}",
                "hourly": data.get("hourly", {})
            })

        elif forecast_type == "1_week":
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                end_dt = start_dt + timedelta(days=6)
                date_param = f"&start_date={start_dt.strftime('%Y-%m-%d')}&end_date={end_dt.strftime('%Y-%m-%d')}"
            else:
                date_param = "&forecast_days=7"
            url = f"{base_url}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum{date_param}"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "temperature_unit": unit_param.capitalize(),
                "timescale": f"7-Day Forecast (Starting {start_date if start_date else 'today'})",
                "daily": data.get("daily", {})
            })

        elif forecast_type == "0_5_month":
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                end_dt = start_dt + timedelta(days=15)
                date_param = f"&start_date={start_dt.strftime('%Y-%m-%d')}&end_date={end_dt.strftime('%Y-%m-%d')}"
            else:
                date_param = "&forecast_days=16"
            url = f"{base_url}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum{date_param}"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "temperature_unit": unit_param.capitalize(),
                "timescale": f"16-Day Extended Forecast (Starting {start_date if start_date else 'today'})",
                "daily": data.get("daily", {})
            })
        else:
            url = f"{base_url}&current_weather=true"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "temperature_unit": unit_param.capitalize(),
                "current": data.get("current_weather", {})
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- 5. REGISTER BOTH API & CHROMADB RAG TOOLS ---
available_tools = {
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real weather forecasts for any city worldwide across different time horizons and temperature units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "forecast_type": {
                        "type": "string",
                        "enum": ["current", "hourly", "1_week", "0_5_month"],
                        "description": "The forecast timescale: 'current', 'hourly', '1_week', or '0_5_month'."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "temperature_unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature scale to use: 'celsius' (°C) or 'fahrenheit' (°F)."
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "ChromaDB RAG tool: Search local .txt files from ./knowledge_base directory containing travel advisories, flight rules, and weather safety policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "City name, country, or weather policy query topic"}
                },
                "required": ["query"],
            },
        },
    }
]


# --- 6. MODEL FALLBACK HELPER ---
def call_llm_with_fallback(client, messages, tools_schema, primary_model="gpt-5.5", fallback_model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=primary_model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        return response, primary_model
    except Exception as e:
        st.warning(f"⚠️ Primary model ({primary_model}) encountered an issue: {e}. Falling back to {fallback_model}...")
        response = client.chat.completions.create(
            model=fallback_model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        return response, fallback_model


# --- 7. MAIN APPLICATION UI ---
st.markdown('<h1 class="main-header">🌤️ Global Weather Intelligence Agent</h1>', unsafe_allow_html=True)
st.caption(f"Powered by an agent harness with real-time weather APIs & persistent ChromaDB vector RAG")

# Disclaimer Banner
st.markdown(
    '''
    <div class="disclaimer-banner">
        ⚠️ <b>Prototype Disclaimer:</b> This application is strictly an experimental demonstration of AI agent orchestration (API + ChromaDB RAG). It does not provide professional travel, safety, or weather advisory services.
    </div>
    ''',
    unsafe_allow_html=True
)

tab_app, tab_explanation = st.tabs(["🚀 Interactive Application", "📖 Harness Architecture Deep Dive"])

# --- TAB 1: INTERACTIVE APPLICATION ---
with tab_app:
    st.subheader("📋 Configure Forecast & Advisory Search")

    if "city_input_text" not in st.session_state:
        st.session_state["city_input_text"] = "Tokyo"

    col1, spacer_col1, col2 = st.columns([1.2, 0.25, 1.1])

    with col1:
        def update_city_from_pill():
            if st.session_state.get("suggestion_pills"):
                st.session_state["city_input_text"] = st.session_state["suggestion_pills"]

        st.markdown('<div class="form-step-label"><span class="form-step-number">1</span> Which city\'s weather are you interested in?</div>', unsafe_allow_html=True)
        raw_city = st.text_input("City", key="city_input_text", label_visibility="collapsed")

        suggestions = search_cities(raw_city)
        if suggestions:
            st.caption("👇 Search a city and pick from a result or suggestion below:")
            st.pills(label="Suggestions", options=suggestions, label_visibility="collapsed", key="suggestion_pills", on_change=update_city_from_pill)

    with col2:
        st.markdown('<div class="form-step-label"><span class="form-step-number">2</span> Query Horizon</div>', unsafe_allow_html=True)
        timescale_choice = st.radio("Horizon", ["Hourly Details", "1-Week Outlook (7 Days)", "0.5 Month Forecast (16 Days)"], label_visibility="collapsed")

    st.markdown("---")

    date_col1, spacer_col2, unit_col2 = st.columns([1.2, 0.45, 1.0])
    
    with date_col1:
        today = date.today()
        max_forecast_date = today + timedelta(days=15)
        
        if "Hourly" in timescale_choice:
            timescale_option = "Hourly"
            st.markdown('<div class="form-step-label"><span class="form-step-number">3</span> Target Date</div>', unsafe_allow_html=True)
            selected_date = st.date_input("Target Date", value=today, min_value=today, max_value=max_forecast_date, label_visibility="collapsed")
            st.caption(f"ℹ️ **Guidance:** Select between **{today.strftime('%b %d')}** and **{max_forecast_date.strftime('%b %d, %Y')}**.")
        elif "1-Week" in timescale_choice:
            timescale_option = "1-Week (7 Days)"
            st.markdown('<div class="form-step-label"><span class="form-step-number">3</span> Start Date</div>', unsafe_allow_html=True)
            max_start_date_7d = today + timedelta(days=9)
            selected_date = st.date_input("Start Date", value=today, min_value=today, max_value=max_start_date_7d, label_visibility="collapsed")
            st.caption(f"ℹ️ **Guidance:** 7-day window runs from **{selected_date.strftime('%b %d')}** to **{(selected_date + timedelta(days=6)).strftime('%b %d, %Y')}**.")
        else:
            timescale_option = "0.5 Month (16 Days)"
            st.markdown('<div class="form-step-label"><span class="form-step-number">3</span> Start Date</div>', unsafe_allow_html=True)
            selected_date = st.date_input("Start Date", value=today, min_value=today, max_value=today, disabled=True, label_visibility="collapsed")
            st.caption(f"ℹ️ **Guidance:** 16-day forecast covers **{today.strftime('%b %d')}** through **{max_forecast_date.strftime('%b %d, %Y')}**.")

    with unit_col2:
        st.markdown('<div class="form-step-label"><span class="form-step-number">4</span> Temperature Unit</div>', unsafe_allow_html=True)
        unit_choice = st.radio("Temperature Scale", ["Celsius (°C)", "Fahrenheit (°F)"], horizontal=True, label_visibility="collapsed")
        chosen_unit = "fahrenheit" if "Fahrenheit" in unit_choice else "celsius"

    st.markdown("<br>", unsafe_allow_html=True)
    include_rag_check = st.checkbox("🔍 Also search internal `./knowledge_base` documents (Triggers ChromaDB RAG Tool)", value=True)

    submit_clicked = st.button("Submit & Run Agent 🚀", type="primary")

    if submit_clicked and raw_city.strip():
        if not api_key:
            st.error("⚠️ OpenAI API Key missing! Set `OPENAI_API_KEY` in `.streamlit/secrets.toml` or export it in your terminal environment variables.")
            st.stop()

        client = OpenAI(api_key=api_key)
        date_str = selected_date.strftime("%Y-%m-%d") if selected_date else today.strftime("%Y-%m-%d")

        unit_label = "Fahrenheit (°F)" if chosen_unit == "fahrenheit" else "Celsius (°C)"

        if timescale_option == "Hourly":
            user_query = f"Give me an hourly weather forecast for {raw_city.strip()} on {date_str} in {unit_label}."
        elif timescale_option == "1-Week (7 Days)":
            end_date_str = (selected_date + timedelta(days=6)).strftime("%Y-%m-%d")
            user_query = f"Give me a 1-week weather forecast for {raw_city.strip()} starting from {date_str} to {end_date_str} in {unit_label}."
        else:
            end_date_str = (selected_date + timedelta(days=15)).strftime("%Y-%m-%d")
            user_query = f"Give me a 0.5 month (16 days) weather forecast for {raw_city.strip()} starting from {date_str} to {end_date_str} in {unit_label}."

        if include_rag_check:
            user_query += f" Also search our knowledge base in ChromaDB for any relevant flight policies or weather warnings for {raw_city.strip()}."

        user_query += f" Always pass `temperature_unit='{chosen_unit}'` to the weather tool, format forecast data in a clear Markdown table, and use identical H3 (###) headers for all section titles."

        st.markdown("---")
        st.markdown(
            f'''
            <div style="background-color: #161b26; border-left: 4px solid #6366f1; padding: 12px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Constructed Agent Query</span><br>
                <span style="color: #c084fc; font-size: 1.1rem; font-weight: 500;">"{user_query}"</span>
            </div>
            ''', 
            unsafe_allow_html=True
        )

        col_trace, col_output = st.columns([1, 1])

        system_instruction = (
            "You are a weather & travel assistant. Use available tools (weather API and ChromaDB RAG internal knowledge base) "
            "to answer user questions completely. Ensure temperatures are displayed in " + unit_label + ". "
            "CRITICAL FORMATTING REQUIREMENT: Strictly use H3 level Markdown headers (###) for ALL section titles in your response "
            "(e.g., '### Weather Forecast', '### Relevant Travel Advisories / Weather Warnings', '### Summary'). "
            "Do NOT mix heading levels (such as ## or ####) or use bold text as standard section titles."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query}
        ]
        
        max_steps = 5
        step = 0

        with col_trace:
            st.write("### ⚙️ Harness Execution Trace")
            with st.spinner("Agent evaluating tools..."):
                while step < max_steps:
                    step += 1
                    response, model_used = call_llm_with_fallback(client, messages, tools_schema, primary_model="gpt-5.5", fallback_model="gpt-4o-mini")
                    response_message = response.choices[0].message
                    messages.append(response_message)

                    tool_calls = response_message.tool_calls
                    if not tool_calls:
                        st.success(f"✅ **Execution Completed** (`{model_used}`)")
                        final_content = response_message.content
                        break

                    for idx, tool_call in enumerate(tool_calls, start=1):
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)
                        
                        tool_type = "ChromaDB RAG Search" if fn_name == "search_knowledge_base" else "Live API Request"
                        badge_color = "#854d0e" if fn_name == "search_knowledge_base" else "#312e81"

                        st.markdown(f"""
                        <div class="harness-card">
                            <span class="harness-badge" style="background-color: {badge_color};">Step {step}.{idx} | {tool_type}</span><br><br>
                            <b>Target Function:</b> <code>{fn_name}</code><br>
                            <b>Arguments:</b> <code>{json.dumps(fn_args)}</code>
                        </div>
                        """, unsafe_allow_html=True)

                        tool_fn = available_tools[fn_name]
                        tool_output = tool_fn(**fn_args)

                        st.markdown(f"""
                        <div class="harness-card" style="border-left-color: #10b981;">
                            <span class="harness-badge" style="background-color: #064e3b; color: #a7f3d0;">Step {step}.{idx} | Result Payload</span><br><br>
                            <code>{tool_output[:220]}...</code>
                        </div>
                        """, unsafe_allow_html=True)

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": tool_output,
                        })

        with col_output:
            st.write("### 📊 Agent Final Response")
            st.markdown(final_content)

# --- TAB 2: HARNESS ARCHITECTURE DEEP DIVE ---
with tab_explanation:
    st.subheader("🏗️ Agent Harness & ChromaDB RAG Architecture Deep Dive")
    
    img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
    with img_col2:
        try:
            st.image("agent-harness.svg", width=650)
        except Exception:
            st.info("💡 Place 'agent-harness.svg' in your working directory to display the flowchart.")

    st.markdown("---")

    st.subheader("📌 System Architecture Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px;">1. Input Prompt Formatting</h4>
        <p style="margin-top: 0;">The user's UI selections are deterministically compiled into a unified prompt string:<br>
        <code>"User choice -> formatted -> Prompt: 'Give me a 1-week weather forecast for Tokyo in Fahrenheit (°F)'"</code></p>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">2. Context & Memory Manager</h4>
        <p style="margin-top: 0;">Maintains the <code>messages</code> history array across execution cycles, preserving system instructions, user queries, and previous tool outputs.</p>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">3. Tool Schema Registry</h4>
        <p style="margin-top: 0;">Exposes executable python tools to the LLM using standard OpenAI JSON schema format:</p>
        <ul>
            <li><b>API Tool</b>: <code>get_weather(location, forecast_type, start_date, temperature_unit)</code></li>
            <li><b>ChromaDB RAG Tool</b>: <code>search_knowledge_base(query)</code></li>
        </ul>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px;">4. LLM Decision & Branch Routing</h4>
        <p style="margin-top: 0;">The inference model inspects the prompt against advertised schemas and routes execution along one or more paths:</p>
        <ul>
            <li><b>Weather API Path</b>: Fetches real-time weather metrics via the Open-Meteo REST API using chosen unit (Celsius/Fahrenheit).</li>
            <li><b>ChromaDB RAG Path</b>: Embeds the query locally and retrieves document matches from persistent disk-backed storage (<code>./chroma_db</code>).</li>
        </ul>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">5. ReAct Execution Loop & Exit</h4>
        <p style="margin-top: 0;">Tool output payloads are appended back to the harness context as <code>{role: "tool"}</code> messages. The harness loops back to the LLM until all function calls complete, producing a synthesized final Markdown response.</p>
        """, unsafe_allow_html=True)