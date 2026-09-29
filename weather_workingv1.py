import json
import os
import requests
from datetime import datetime
from openai import OpenAI

# Safe API Key initialization
api_key = os.environ.get("OPENAI_API_KEY")

try:
    import streamlit as st
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass

client = OpenAI(api_key=api_key)


# --- 1. HARNESS EXPLANATION ---

EXPLANATION = """
===============================================================================
                          WHAT IS THE AGENT HARNESS?
===============================================================================
The Large Language Model (LLM) itself is just a passive text generator. 
The "Harness" is the surrounding Python runtime environment that turns 
the LLM into an active, tool-using agent.

This script contains 4 key components that make up the harness:

1. THE CONTEXT MEMORY PIPELINE (`messages = [...]`)
   - Keeps track of the entire conversation across steps.

2. THE TOOL REGISTRY & DISPATCHER (`available_tools` + `tools_schema`)
   - Maps LLM tool choices directly to executable Python code.

3. THE REASON-ACT LOOP (`while step < max_steps:`)
   - Automated multi-step execution loop:
     Query -> Model Decides Tool -> Harness Runs Tool -> Result Back -> Final Answer

4. SAFETY GUARDRAILS (`max_steps = 5`)
   - Prevents infinite loops and controls API costs.
===============================================================================
"""


# --- 2. TOOL FUNCTION IMPLEMENTATION ---

def get_weather(location: str, forecast_type: str = "current", target_date: str = None) -> str:
    """
    Fetches weather forecasts for any city worldwide using Open-Meteo API.
    Supports current, hourly (for a specific date YYYY-MM-DD), 1_week, or 1_month (16 days).
    """
    try:
        # Step A: Convert city name to coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return json.dumps({"error": f"Location '{location}' not found."})

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        city_name = geo_res["results"][0]["name"]
        country = geo_res["results"][0].get("country", "")

        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&timezone=auto"

        # Step B: Fetch requested forecast timescale
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


# --- 3. HARNESS REGISTRIES & SCHEMAS ---

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
                    "location": {
                        "type": "string", 
                        "description": "The city name (e.g. Tokyo, London, Paris)"
                    },
                    "forecast_type": {
                        "type": "string",
                        "enum": ["current", "hourly", "1_week", "1_month"],
                        "description": "The time horizon: 'current', 'hourly', '1_week', or '1_month'."
                    },
                    "target_date": {
                        "type": "string",
                        "description": "The target date in YYYY-MM-DD format (used when forecast_type is hourly)."
                    }
                },
                "required": ["location"],
            },
        },
    }
]


# --- 4. THE HARNESS CONTROL LOOP ---

def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant with access to real-time global weather tools. Summarize forecast outputs clearly with key numbers."},
        {"role": "user", "content": user_query}
    ]
    
    max_steps = 5  # Harness Guardrail
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n--- Harness Loop Step {step} ---")
        
        # Call LLM Brain
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)  # Save context memory

        # Check for tool call execution requests
        tool_calls = response_message.tool_calls
        if not tool_calls:
            print("\nFinal Answer:")
            print(response_message.content)
            break

        # Execute requested tools via dispatcher
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Executing Tool: {function_name}({function_args})")
            
            # Execute local Python function
            tool_function = available_tools[function_name]
            tool_output = tool_function(**function_args)

            # Append execution output back into memory
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_output,
            })


# --- 5. INTERACTIVE USER INPUT MENU ---

def prompt_user_and_run():
    print(EXPLANATION)
    print("\n=== GLOBAL WEATHER AGENT HARNESS ===")

    while True:
        print("\n" + "="*40)
        city = input("Enter City Name (or type 'exit' to quit): ").strip()
        if city.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not city:
            continue

        print("\nSelect Forecast Horizon:")
        print("1. Hourly")
        print("2. 1-Week (7 days)")
        print("3. 1-Month (16 days extended)")
        
        choice = input("Enter option (1, 2, or 3): ").strip()

        target_date = None
        if choice == "1":
            forecast_type = "hourly"
            print("\nEnter Date for Hourly Forecast (YYYY-MM-DD)")
            date_input = input("Date [Press ENTER for Today]: ").strip()
            if date_input:
                target_date = date_input
            else:
                target_date = datetime.now().strftime("%Y-%m-%d")

            query = f"Give me an hourly weather forecast for {city} on {target_date}."

        elif choice == "2":
            forecast_type = "1_week"
            query = f"Give me a 1-week weather forecast for {city}."

        elif choice == "3":
            forecast_type = "1_month"
            query = f"Give me a 1-month weather forecast for {city}."

        else:
            print("Invalid selection. Defaulting to 1-week forecast.")
            query = f"Give me a 1-week weather forecast for {city}."

        print(f"\nConstructed Prompt for Agent: '{query}'")
        run_agent(query)


if __name__ == "__main__":
    prompt_user_and_run()