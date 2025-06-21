import os
import requests
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent

load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

import google.generativeai as genai
genai.configure(api_key=genai_api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

search = DuckDuckGoSearchRun()

templates = {
    "train-info": PromptTemplate(
        input_variables=["Departure_City", "Destination_City", "Budget_Amount"],
        template="""
        Fetch travel info from {Departure_City} to {Destination_City} under {Budget_Amount} budget.
        1. Distance in km.
        2. 2-3 train ticket options in table:
        | Train Name | Train Number | Departure | Arrival | Sleeper | 3AC | 2AC |
    """
    ),
    "flight_info": PromptTemplate(
        template="""
        Provide flight options from {Departure_City} to {Destination_City} under {Budget_Amount} budget.

        Show only top 2-3 relevant options in the table below:

        | Airline | Flight No. | Departure Time | Arrival Time | Duration | Price (₹) | Booking Link |
        | ------- | ---------- | --------------- | ------------- | -------- | ---------- | ------------- |

        - Ensure price is in rupees.
        - Only show budget-friendly options that fall within the specified range.
        - If booking link not available, mention "https://www.google.com/flights".
        - Ensure the flights are realistic and commonly available.
        """
    ),
    "hotels_food_places": PromptTemplate(
        input_variables=["Destination_City", "Budget_Amount"],
        template="""
        List top 2-3 budget hotels, local foods, and attractions in {Destination_City}.
        | Hotel Name | Price/Night | Location | Rating ⭐ | Booking Link |
        | Dish | Description |
        | Place Name | Description | Visit Duration | Entry Fee |
        """
    ),
    "itinerary": PromptTemplate(
        input_variables=["Destination_City", "Travel_Days", "Interest_Type"],
        template="""
        Create a {Travel_Days}-day itinerary for {Destination_City} focused on {Interest_Type}.
        | Day | Morning | Afternoon | Evening | Duration |
        """
    ),
    "weather": PromptTemplate(
        input_variables=["Destination_City", "Travel_Dates"],
        template="""
        Weather forecast for {Destination_City} from {Travel_Dates}.
        | Date | Morning Temp | Afternoon Temp | Evening Temp | Conditions |
        """
    ),
    "transport": PromptTemplate(
        input_variables=["Destination_City"],
        template="""
        Local transport options in {Destination_City}.
        | Mode | Cost | Availability | Duration |
        """
    ),
    "festivals": PromptTemplate(
        input_variables=["Destination_City", "Travel_Dates"],
        template="""
        Festivals/events in {Destination_City} during {Travel_Dates}.
        | Event Name | Description | Date & Time | Venue |
        """
    ),
    "travel_tips": PromptTemplate(
        input_variables=["Destination_City"],
        template="""
        Travel safety tips for {Destination_City}.
        | Tip Type | Details |
        """
    ),
    "packing_list": PromptTemplate(
        input_variables=["Destination_City", "Travel_Days", "Planned_Activities"],
        template="""
        Packing list for {Travel_Days}-day trip to {Destination_City} for {Planned_Activities}.
        | Item | Quantity |
        """
    )
}
def get_weather_forecast(city, travel_dates):
    try:
        start_date, end_date = [x.strip() for x in travel_dates.split("to")]
        location_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        loc_response = requests.get(location_url).json()
        lat = loc_response['results'][0]['latitude']
        lon = loc_response['results'][0]['longitude']

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone=auto"
        )
        weather_data = requests.get(weather_url).json()
        days = weather_data["daily"]["time"]
        temp_max = weather_data["daily"]["temperature_2m_max"]
        temp_min = weather_data["daily"]["temperature_2m_min"]
        weathercodes = weather_data["daily"]["weathercode"]

        weather_conditions = {
            0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Rime fog", 51: "Drizzle", 61: "Light rain",
            71: "Snow", 95: "Thunderstorm"
        }

        result = "| Date | Morning Temp | Afternoon Temp | Evening Temp | Conditions |\n"
        result += "|------|---------------|-----------------|----------------|-------------|\n"
        for i in range(len(days)):
            condition = weather_conditions.get(weathercodes[i], "Unknown")
            result += f"| {days[i]} | {temp_min[i]}°C | {temp_max[i]}°C | {temp_min[i]+2}°C | {condition} |\n"
        return result
    except Exception as e:
        return f"❌ Error fetching weather: {e}"

weather_tool = Tool(
    name="WeatherForecastTool",
    func=lambda x: get_weather_forecast(x["Destination_City"], x["Travel_Dates"]),
    description="Returns weather forecast for a given city and date range."
)

agents = {
    key: initialize_agent(
        tools=[search] if key != "weather" else [weather_tool],
        llm=llm,
        agent_type="structured-chat-zero-shot-react-description",
        verbose=False,
        handle_parsing_errors=True
    )
    for key in templates.keys()
}

def get_travel_info(category, **kwargs):
    if category == "weather":
        return get_weather_forecast(kwargs["Destination_City"], kwargs["Travel_Dates"])
    else:
        prompt_text = templates[category].format(**kwargs)
        response = agents[category].invoke({"input": prompt_text})
        return response.get("output", "Error: No output received.")

st.title("AI-Powered Travel Planner 🧳")

Departure_City = st.text_input("🏙 Departure City")
Destination_City = st.text_input("📍 Destination City")
Budget_Amount = st.number_input("💰 Budget Amount", min_value=0, step=500)
Travel_Days = st.number_input("📅 Number of Days", min_value=1, step=1)
Interest_Type = st.selectbox("🎯 Interest Type", ["Adventure", "Relaxation", "Sightseeing", "Food & Drinks"])
Planned_Activities = st.text_area("🧳 Planned Activities")
Travel_Dates = st.text_input("📆 Travel Dates (YYYY-MM-DD to YYYY-MM-DD)")

selected_category = st.selectbox(
    "Select travel info category:",
    options=list(templates.keys())
)

if st.button("Get Travel Plan"):
    with st.spinner("Fetching travel details..."):
        if selected_category == "train-info":
            output = get_travel_info(
                selected_category,
                Departure_City=Departure_City,
                Destination_City=Destination_City,
                Budget_Amount=Budget_Amount
            )
        elif selected_category == "flight_info":
            output = get_travel_info(
                selected_category,
                Departure_City=Departure_City,
                Destination_City=Destination_City,
                Budget_Amount=Budget_Amount
            )
        elif selected_category == "hotels_food_places":
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City,
                Budget_Amount=Budget_Amount
            )
        elif selected_category == "itinerary":
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City,
                Travel_Days=Travel_Days,
                Interest_Type=Interest_Type
            )
        elif selected_category == "weather":
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City,
                Travel_Dates=Travel_Dates
            )
        elif selected_category in ["transport", "travel_tips"]:
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City
            )
        elif selected_category == "festivals":
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City,
                Travel_Dates=Travel_Dates
            )
        elif selected_category == "packing_list":
            output = get_travel_info(
                selected_category,
                Destination_City=Destination_City,
                Travel_Days=Travel_Days,
                Planned_Activities=Planned_Activities
            )
        else:
            output = "No handler implemented for this category."

        st.subheader(selected_category.replace("_", " ").title())
        st.markdown(output)
