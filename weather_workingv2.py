import json
import os
import requests
from datetime import datetime, date
import streamlit as st
from openai import OpenAI

# Page Configuration & Styling
st.set_page_config(
    page_title="Global Weather Agent Harness",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI accents
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e3440;
        margin-bottom: 10px;
    }
    .harness-step {
        border-left: 4px solid #4f46e5;
        background-color: #1a1d24;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. API KEY INITIALIZATION ---
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API Key missing! Set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

client = OpenAI(api_key=api_key)


# --- 2. TOOL FUNCTION IMPLEMENTATION ---
def get_weather(location: str, forecast_type: str = "current", target_date: str = None) -> str:
    """Fetches real-time, hourly, 1-week, or 1-month forecasts via Open-Meteo API."""
    try:
        # Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return json.dumps({"error": f"Location '{location}' not found."})

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        city_name = geo_res["results"][0]["name"]
        country = geo_res["results"][0].get("country", "")

        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&timezone=auto"

        if forecast_type == "hourly":
            date_param = f"&start_date={target_date}&end_date={target_date}" if target_date else "&forecast_days=1"
            url = f"{base_url}&hourly=temperature_2m,precipitation_probability{date_param}"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "timescale": f"Hourly forecast for {target_date if target_date else 'today'}",
                "hourly": data.get("hourly", {})
            })

        elif forecast_type == "1_week":
            url = f"{base_url}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=7"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "timescale": "7-Day Daily Forecast",
                "daily": data.get("daily", {})
            })

        elif forecast_type == "1_month":
            url = f"{base_url}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=16"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "timescale": "16-Day Extended Forecast",
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


# --- 3. TOOL REGISTRY & SCHEMAS ---
available_tools = {"get_weather": get_weather}

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
                        "enum": ["current", "hourly", "1_week", "1_month"],
                        "description": "The forecast timescale"
                    },
                    "target_date": {
                        "type": "string",
                        "description": "Target date in YYYY-MM-DD format (for hourly forecast)"
                    }
                },
                "required": ["location"],
            },
        },
    }
]


# --- 4. STREAMLIT UI LAYOUT ---

st.title("🌤️ Global Weather Agent Harness")
st.caption("Powered by LLM Function Calling + Open-Meteo Runtime Infrastructure")

# Layout Tabs
tab_app, tab_explanation = st.tabs(["🚀 Interactive App", "📖 How the Harness Works"])

with tab_explanation:
    st.subheader("What is an Agent Harness?")
    st.markdown("""
    An **agent harness** is the surrounding software infrastructure that turns a raw, passive Large Language Model (LLM) into an active, tool-using system capable of solving multi-step tasks.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🧠 **LLM (The Brain)**\nProcesses text, determines intent, and decides *which* tool to run with *what* parameters.")
        st.success("⚙️ **Harness Loop (Execution)**\nAutomates the `Query → Decision → Tool Execution → Result → Response` cycle.")
    with col2:
        st.warning("💾 **Memory & State**\nMaintains conversation state and feeds API responses back into context.")
        st.error("🛡️ **Safety Guardrails**\nLimits total steps (`max_steps`) to prevent infinite execution loops.")

with tab_app:
    # Sidebar Controls
    st.sidebar.header("🕹️ Forecast Controls")
    
    city = st.sidebar.text_input("Enter City Name", value="Tokyo")
    
    timescale_option = st.sidebar.selectbox(
        "Select Time Horizon",
        ["Hourly", "1-Week (7 Days)", "1-Month (16 Days)"]
    )
    
    selected_date = None
    if timescale_option == "Hourly":
        selected_date = st.sidebar.date_input("Select Date for Hourly Forecast", value=date.today())

    fetch_button = st.sidebar.button("Run Weather Agent", type="primary", use_container_width=True)

    if fetch_button and city:
        # Construct Prompt based on UI inputs
        if timescale_option == "Hourly":
            date_str = selected_date.strftime("%Y-%m-%d")
            user_query = f"Give me an hourly weather forecast for {city} on {date_str}."
            forecast_type = "hourly"
        elif timescale_option == "1-Week (7 Days)":
            user_query = f"Give me a 1-week weather forecast for {city}."
            forecast_type = "1_week"
        else:
            user_query = f"Give me a 1-month weather forecast for {city}."
            forecast_type = "1_month"

        st.subheader(f"Query: \"{user_query}\"")
        
        # Display Columns: Left = Harness Execution Trace, Right = Final Result
        col_trace, col_output = st.columns([1, 1])

        messages = [
            {"role": "system", "content": "You are a helpful weather assistant with access to real-time global weather tools. Summarize outputs cleanly with metrics."},
            {"role": "user", "content": user_query}
        ]
        
        max_steps = 5
        step = 0

        with col_trace:
            st.write("### ⚙️ Harness Execution Loop")
            
            with st.spinner("Agent running..."):
                while step < max_steps:
                    step += 1
                    
                    # LLM Brain Call
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=tools_schema,
                        tool_choice="auto"
                    )
                    
                    response_message = response.choices[0].message
                    messages.append(response_message)

                    tool_calls = response_message.tool_calls
                    
                    if not tool_calls:
                        st.success("✅ Execution complete: Final response synthesized.")
                        final_content = response_message.content
                        break

                    for tool_call in tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)
                        
                        st.markdown(f"""
                        <div class="harness-step">
                            <b>Step {step}: LLM Tool Call Requested</b><br>
                            <code>Function: {fn_name}</code><br>
                            <code>Args: {json.dumps(fn_args)}</code>
                        </div>
                        """, unsafe_allow_html=True)

                        # Execute Tool
                        tool_fn = available_tools[fn_name]
                        tool_output = tool_fn(**fn_args)

                        # Append back to memory
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": tool_output,
                        })

        with col_output:
            st.write("### 📊 Agent Final Response")
            st.write(final_content)