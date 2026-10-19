import uuid
import streamlit as st

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
st.title("✈️ AI Multi-Agent Travel Planner")
st.caption("Research-backed itinerary planning. Flight and price details are only shown as verified when supported by providers.")

with st.sidebar:
    st.subheader("Before you start")
    st.markdown(
        "- Include origin, destination, exact dates, traveler count, and interests.\n"
        "- Search results are research leads, not bookings.\n"
        "- Verify prices, availability, and opening hours with providers."
    )

user_query = st.text_area(
    "Describe your trip",
    placeholder="Plan a 3-day trip from Delhi to Dubai from 2026-11-10 to 2026-11-12 for 2 travelers. Include sightseeing, food, and hotels.",
    height=140,
)
run = st.button("Generate travel plan", type="primary", use_container_width=True)

if run:
    if not user_query.strip():
        st.warning("Please describe your trip first.")
        st.stop()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {"user_query": user_query.strip(), "limitations": [], "llm_calls": 0}
    updates = {}
    progress = st.status("Researching your trip and preparing the itinerary…", expanded=True)
    try:
        from main import app

        for update in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in update.items():
                if isinstance(node_state, dict):
                    updates.update(node_state)
                    progress.write(f"Completed: {node_name.replace('_', ' ').title()}")
        progress.update(label="Travel plan generated", state="complete", expanded=False)
    except Exception as exc:
        progress.update(label="Travel plan could not be completed", state="error", expanded=True)
        st.error(f"Something went wrong ({type(exc).__name__}).")
        st.info("Check GROQ_API_KEY, DATABASE_URL, package installation, and API configuration.")
        st.stop()

    final_response = updates.get("final_response", "")
    if final_response:
        st.markdown(final_response)
        st.download_button("Download travel plan (.md)", data=final_response, file_name="travel_plan.md", mime="text/markdown")
    else:
        st.warning("The graph finished without a final response. Check the terminal logs.")
    with st.expander("Debug details"):
        st.write("LLM calls:", updates.get("llm_calls", 0))
        st.write("Flight result:", updates.get("flight_results", "Not returned"))
        st.write("Limitations:", updates.get("limitations", []))
