import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


search = DuckDuckGoSearchRun()

templates = {
    "distance_train": PromptTemplate(
        input_variables=["Departure_City", "Destination_City", "Budget_Amount"],
        template="""
        Fetch verified data for travel from {Departure_City} to {Destination_City} on a budget of {Budget_Amount}.

        **1. Distance**
        - Exact distance in km.

        **2. Train Ticket Options**
        - Provide details for up to **2-3** relevant train options in the following table:
        - give details to below following table....
        | Train Name | Train Number | Departure | Arrival | Sleeper | 3AC | 2AC |
        | --- | --- | --- | --- | --- | --- | --- |
        - give relevant price in rupees
        - Only **top 2-3 relevant trains** should be included.
        - Ensure accurate pricing and booking links
        """
    ),
    "hotels_food_places": PromptTemplate(
        input_variables=["Destination_City", "Budget_Amount"],
        template="""
        Provide **top 2-3 budget hotels**, famous local foods, and must-visit places in {Destination_City}.
        
        **1. Hotels** (within budget)
        List **only 2-3 best hotels** within the budget, formatted as:
        - ensure that budget amount for Travel_Days...
        - ensure that prices are in rupees
        | Hotel Name | Price/Night | Location | Rating ⭐ | Booking Link |
        | --- | --- | --- | --- | --- |
        - if the hotel booking website is not available then mention (https://www.makemytrip.com/hotels/) this website.
        - If specific hotel prices are unavailable, provide general price ranges

        **2. Local Foods**
        | Dish | Description |
        | --- | ----------- |

        **3. Famous Places to Visit**
        Provide **only 4-5** key attractions in a structured table:
        | Place Name | Description | Visit Duration | Entry Fee |
        | --- | --- | --- | --- |
        """
    ),
    "itinerary": PromptTemplate(
        input_variables=["Destination_City", "Travel_Days", "Interest_Type"],
        template="""
        Create a {Travel_Days}-day itinerary for {Destination_City} focused on {Interest_Type}.
        - use this type of prompt to structual format...
        | Day | Morning | Afternoon | Evening | Duration |
        | --- | ------- | --------- | ------- | -------- |
        """
    ),
    "weather": PromptTemplate(
        input_variables=["Destination_City", "Travel_Dates"],
        template="""
        Provide a weather forecast for {Destination_City} from {Travel_Dates}.
        - use this type of prompt to structual format...
        - if can you not provide proper information then mention not provide wrong information.
        | Date | Morning Temp | Afternoon Temp | Evening Temp | Conditions |
        | ---- | ------------ | -------------- | ------------ | ---------- |
        """
    ),
    "transport": PromptTemplate(
        input_variables=["Destination_City"],
        template="""
        Provide local transport options for {Destination_City}.
        - use this type of prompt to structual format...
        | Mode | Cost | Availability | Duration |
        | ---- | ---- | ------------ | -------- |
        """
    ),
    "festivals": PromptTemplate(
        input_variables=["Destination_City", "Travel_Dates"],
        template="""
        List events and festivals in {Destination_City} during {Travel_Dates}.
        if not any events and festival then mention not give wrong information.
        | Event Name | Description | Date & Time | Venue |
        | ---------- | ----------- | ----------- | ----- |
        """
    ),
    "travel_tips": PromptTemplate(
        input_variables=["Destination_City"],
        template="""
        Provide travel safety tips and alerts for {Destination_City}.

        | Tip Type | Details |
        | -------- | ------- |
        | Safety   | [Safety Tip] |
        | Visa     | [Visa Details] |
        | COVID-19 | [Guidelines] |
        """
    ),
    "packing_list": PromptTemplate(
        input_variables=["Destination_City", "Travel_Days", "Planned_Activities"],
        template="""
        Generate a packing list for a {Travel_Days}-day trip to {Destination_City} considering {Planned_Activities}.

        **Packing List**
        | Item | Quantity |
        | ---- | -------- |
        | Lightweight clothes | 5 |
        | Walking shoes | 2 pairs |
        | Travel documents | 1 set |
        """
    )
}


agents = {
    key: initialize_agent(
        [search], llm,
        agent_type="structured-chat-zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )
    for key in templates.keys()
}


def get_travel_info(category, **kwargs):
    prompt_text = templates[category].format(**kwargs)
    inputs = {"input": prompt_text}
    response = agents[category].invoke(inputs)
    return response.get("output", "Error: No output received.")

st.title("AI-Powered Travel Planner 🧳")

Departure_City = st.text_input("🏙️ Departure City")
Destination_City = st.text_input("📍 Destination City")
Budget_Amount = st.number_input("💰 Budget Amount", min_value=0, step=500)
Travel_Days = st.number_input("📅 Number of Days", min_value=1, step=1)
Interest_Type = st.selectbox("🎯 Interest Type", ["Adventure", "Relaxation", "Sightseeing", "Food & Drinks"])
Planned_Activities = st.text_area("🧳 Planned Activities")
Travel_Dates = st.text_input("📆 Travel Dates (YYYY-MM-DD to YYYY-MM-DD)")

category_options = list(templates.keys())
selected_categories = st.multiselect(
    "Select travel info categories you want:",
    options=category_options,
    default=["distance_train", "hotels_food_places"]
)

if st.button("Get Travel Plan"):
    if not selected_categories:
        st.warning("Please select at least one category.")
    else:
        with st.spinner("Fetching travel details..."):
            results = {}
            for category in selected_categories:
                # Pass parameters according to each prompt template
                if category == "distance_train":
                    results[category] = get_travel_info(
                        category,
                        Departure_City=Departure_City,
                        Destination_City=Destination_City,
                        Budget_Amount=Budget_Amount
                    )
                elif category == "hotels_food_places":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City,
                        Budget_Amount=Budget_Amount
                    )
                elif category == "itinerary":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City,
                        Travel_Days=Travel_Days,
                        Interest_Type=Interest_Type
                    )
                elif category == "weather":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City,
                        Travel_Dates=Travel_Dates
                    )
                elif category == "transport":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City
                    )
                elif category == "festivals":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City,
                        Travel_Dates=Travel_Dates
                    )
                elif category == "travel_tips":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City
                    )
                elif category == "packing_list":
                    results[category] = get_travel_info(
                        category,
                        Destination_City=Destination_City,
                        Travel_Days=Travel_Days,
                        Planned_Activities=Planned_Activities
                    )
                else:
                    results[category] = "No handler implemented for this category."

            for key, value in results.items():
                st.subheader(key.replace("_", " ").title())
                st.markdown(value)
