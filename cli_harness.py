import json
import requests
from openai import OpenAI

client = OpenAI()

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
   - Keeps track of the entire conversation.
   - Appends model output AND tool results back into the history array so 
     the stateless LLM maintains context across steps.

2. THE TOOL REGISTRY & DISPATCHER (`available_tools` + `tools_schema`)
   - `tools_schema`: Advertises available functions and arguments to the LLM.
   - `available_tools`: A dictionary mapping schema names directly to executable
     Python functions (`get_weather`).
   - The harness parses the LLM's tool request, runs the corresponding Python 
     function, and returns the raw data.

3. THE REASON-ACT LOOP (`while step < max_steps:`)
   - Automated multi-step execution loop:
     Query -> Model Decides Tool -> Harness Runs Tool -> Result Back to Model -> Final Answer

4. SAFETY GUARDRAILS (`max_steps = 5`)
   - Prevents infinite tool-calling loops and controls token/API cost.
===============================================================================
"""


# --- 2. TOOL FUNCTION IMPLEMENTATION ---

def get_weather(location: str, forecast_type: str = "current") -> str:
    """
    Fetches real-time, 24-hour, 7-day, or extended 16-day weather forecasts 
    for any city worldwide using the Open-Meteo API.
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

        # Step B: Fetch requested timescale
        if forecast_type == "hourly":
            url = f"{base_url}&hourly=temperature_2m,precipitation_probability&forecast_days=1"
            data = requests.get(url).json()
            return json.dumps({
                "location": f"{city_name}, {country}",
                "timescale": "Hourly (24 Hours)",
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
                        "description": "The city name (e.g. Tokyo, London, São Paulo)"
                    },
                    "forecast_type": {
                        "type": "string",
                        "enum": ["current", "hourly", "1_week", "1_month"],
                        "description": "The time horizon: 'current' for instant weather, 'hourly' for next 24 hours, '1_week' for 7-day outlook, or '1_month' for extended forecast."
                    }
                },
                "required": ["location"],
            },
        },
    }
]


# --- 4. THE HARNESS CONTROL LOOP ---

def run_agent(user_query: str):
    # Display the harness explanation header first
    print(EXPLANATION)
    
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant with access to real-time global weather tools."},
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
        messages.append(response_message)  # Save context

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

if __name__ == "__main__":
    run_agent("Give me a 1-week forecast for Tokyo.")
