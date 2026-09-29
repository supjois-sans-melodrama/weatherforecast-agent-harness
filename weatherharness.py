import json
import os
import requests
import numpy as np
from datetime import datetime, date, timedelta
import streamlit as st
from openai import OpenAI

# --- PAGE CONFIGURATION & CUSTOM STYLING ---
st.set_page_config(
    page_title="Global Weather Intelligence | Agent Harness + RAG & API",
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
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
</style>
""", unsafe_allow_html=True)


# --- 1. API KEY INITIALIZATION ---
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API Key missing! Set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

client = OpenAI(api_key=api_key)


# --- 2. VECTOR RAG KNOWLEDGE BASE SETUP ---
SAMPLE_KNOWLEDGE_DOCS = [
    {"id": "doc1", "text": "Travel Advisory Tokyo: Monsoon and rain activity typically spikes during summer months (June to August). Packing light rain gear is recommended."},
    {"id": "doc2", "text": "Travel Advisory Kathmandu: Winter temperatures (November to January) can drop to 2°C at night. Layered clothing and thermal jackets are required."},
    {"id": "doc3", "text": "Flight Cancellation Policy due to Severe Weather: Flights delayed over 4 hours due to storms or typhoons qualify for full ticket refund or free rebooking."},
    {"id": "doc4", "text": "Heatwave Emergency Guidelines: When temperatures exceed 38°C, stay indoors between 12 PM and 4 PM, consume electrolytes, and avoid heavy exercise."}
]

@st.cache_resource
def build_vector_store():
    """Generates embeddings for knowledge base documents using OpenAI text-embedding-3-small."""
    store = []
    for doc in SAMPLE_KNOWLEDGE_DOCS:
        res = client.embeddings.create(input=doc["text"], model="text-embedding-3-small")
        embedding = res.data[0].embedding
        store.append({"doc": doc, "embedding": embedding})
    return store

knowledge_vector_store = build_vector_store()

def search_knowledge_base(query: str) -> str:
    """RAG Tool: Searches internal document store using vector similarity."""
    try:
        q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        scores = []
        for item in knowledge_vector_store:
            dot_product = np.dot(q_emb, item["embedding"])
            norm_q = np.linalg.norm(q_emb)
            norm_doc = np.linalg.norm(item["embedding"])
            sim = dot_product / (norm_q * norm_doc)
            scores.append((sim, item["doc"]["text"]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = [doc_text for sim, doc_text in scores[:2] if sim > 0.3]
        
        if not top_matches:
            return json.dumps({"results": ["No highly relevant local travel/weather advisory documents found."]})
        
        return json.dumps({"retrieved_context": top_matches})
    except Exception as e:
        return json.dumps({"error": f"Failed to execute RAG search: {str(e)}"})


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
def get_weather(location: str, forecast_type: str = "current", start_date: str = None) -> str:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return json.dumps({"error": f"Location '{location}' not found. Please provide a valid city name."})

        result = geo_res["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        city_name, country = result["name"], result.get("country", "")
        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&timezone=auto"

        if forecast_type == "hourly":
            date_param = f"&start_date={start_date}&end_date={start_date}" if start_date else "&forecast_days=1"
            url = f"{base_url}&hourly=temperature_2m,precipitation_probability{date_param}"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
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
                "timescale": f"16-Day Extended Forecast (Starting {start_date if start_date else 'today'})",
                "daily": data.get("daily", {})
            })
        else:
            url = f"{base_url}&current_weather=true"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "current": data.get("current_weather", {})
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- 5. REGISTER BOTH API & RAG TOOLS ---
available_tools = {
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real weather forecasts for any city worldwide across different time horizons.",
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
            "description": "RAG tool: Search internal travel advisories, flight cancellation rules, and heatwave safety policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query topic or question"}
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
st.caption("Powered by an agent harness with real-time weather APIs & vector RAG policy search")

# Disclaimer Banner
st.markdown(
    '''
    <div class="disclaimer-banner">
        ⚠️ <b>Experimental Prototype Disclaimer:</b> This application is strictly an experimental demonstration of AI agent orchestration (API + RAG). It does not provide professional travel, safety, or weather advisory services.
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

    col1, spacer_col, col2 = st.columns([1.2, 0.15, 1.1])

    with col1:
        def update_city_from_pill():
            if st.session_state.get("suggestion_pills"):
                st.session_state["city_input_text"] = st.session_state["suggestion_pills"]

        st.markdown('<div class="form-step-label"><span class="form-step-number">1</span> Which city\'s weather are you interested in?</div>', unsafe_allow_html=True)
        raw_city = st.text_input("City", key="city_input_text", label_visibility="collapsed")

        suggestions = search_cities(raw_city)
        if suggestions:
            st.caption("👇 Type a city of interest or select a suggestion below:")
            st.pills(label="Suggestions", options=suggestions, label_visibility="collapsed", key="suggestion_pills", on_change=update_city_from_pill)

    with col2:
        st.markdown('<div class="form-step-label"><span class="form-step-number">2</span> Query Horizon</div>', unsafe_allow_html=True)
        timescale_choice = st.radio("Horizon", ["Hourly Details", "1-Week Outlook (7 Days)", "0.5 Month Forecast (16 Days)"], label_visibility="collapsed")

    st.markdown("---")
    date_col1, _ = st.columns([1, 1])
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

    include_rag_check = st.checkbox("🔍 Also search internal Travel Advisories & Policies (Triggers RAG Tool)", value=True)

    submit_clicked = st.button("Submit & Run Agent 🚀", type="primary")

    if submit_clicked and raw_city.strip():
        date_str = selected_date.strftime("%Y-%m-%d") if selected_date else today.strftime("%Y-%m-%d")

        if timescale_option == "Hourly":
            user_query = f"Give me an hourly weather forecast for {raw_city.strip()} on {date_str}."
        elif timescale_option == "1-Week (7 Days)":
            end_date_str = (selected_date + timedelta(days=6)).strftime("%Y-%m-%d")
            user_query = f"Give me a 1-week weather forecast for {raw_city.strip()} starting from {date_str} to {end_date_str}."
        else:
            end_date_str = (selected_date + timedelta(days=15)).strftime("%Y-%m-%d")
            user_query = f"Give me a 0.5 month (16 days) weather forecast for {raw_city.strip()} starting from {date_str} to {end_date_str}."

        if include_rag_check:
            user_query += f" Also search our travel advisories for any relevant flight policies or weather warnings for {raw_city.strip()}."

        user_query += " Format forecast data in a clear Markdown table."

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

        messages = [
            {"role": "system", "content": "You are a weather & travel assistant. Use available tools (weather API and RAG internal knowledge base) to answer user questions completely."},
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
                        
                        tool_type = "RAG Vector Search" if fn_name == "search_knowledge_base" else "Live API Request"
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
    st.subheader("🏗️ Agent Harness & RAG Architecture Deep Dive")
    
    # 1. RESIZED & CENTERED DIAGRAM
    img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
    with img_col2:
        try:
            st.image("agent-harness-rag.svg", width=650)
        except Exception:
            st.info("💡 Place 'agent-harness-rag-ocean.svg' in your working directory to display the flowchart.")

    st.markdown("---")

    # 2. ARCHITECTURE SUMMARY EXPLANATION WITH BLUISH-PURPLE HEADINGS
    st.subheader("📌 System Architecture Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px;">1. Input Prompt Formatting</h4>
        <p style="margin-top: 0;">The user's UI selections (city choice, date horizon, and advisory checkboxes) are deterministically compiled into a unified prompt string:<br>
        <code>"User choice -> formatted -> Prompt: 'Give me a 1-week weather forecast for Tokyo'"</code></p>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">2. Context & Memory Manager</h4>
        <p style="margin-top: 0;">Maintains the <code>messages</code> history array across execution cycles, preserving system instructions, user queries, and previous tool outputs.</p>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">3. Tool Schema Registry</h4>
        <p style="margin-top: 0;">Exposes executable python tools to the LLM using standard OpenAI JSON schema format:</p>
        <ul>
            <li><b>API Tool</b>: <code>get_weather(location, forecast_type, start_date)</code></li>
            <li><b>RAG Tool</b>: <code>search_knowledge_base(query)</code></li>
        </ul>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px;">4. LLM Decision & Branch Routing</h4>
        <p style="margin-top: 0;">The inference model (e.g., GPT-5.5) inspects the prompt against advertised schemas and routes execution along one or more paths:</p>
        <ul>
            <li><b>Weather API Path</b>: Fetches real-time weather metrics via the Open-Meteo REST API.</li>
            <li><b>RAG Search Path</b>: Embeds the query and queries internal travel/flight policies using vector cosine similarity.</li>
            <li><b>No Tool Needed Path</b> <i>(Rarely used in this query-driven app)</i>: Handles direct conversational prompts without external calls.</li>
        </ul>

        <h4 style="color: #a78bfa; font-weight: 600; margin-bottom: 8px; margin-top: 20px;">5. ReAct Execution Loop & Exit</h4>
        <p style="margin-top: 0;">Tool output JSON/text payloads are appended back to the harness context as <code>{role: "tool"}</code> messages. The harness loops back to the LLM until all function calls complete, producing a synthesized final Markdown response.<br>
        <i>💡 <b>Note:</b> <b><span style="color: #a78bfa;">Re</span><span style="color: #a78bfa;">Act</span></b> refers to a combination of internal model <b><span style="color: #a78bfa;">Re</span></b>asoning with external tool <b><span style="color: #a78bfa;">Act</span></b>ion invocation.</i></p>
        """, unsafe_allow_html=True)