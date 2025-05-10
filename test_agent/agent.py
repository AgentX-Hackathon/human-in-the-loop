from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import uuid # For unique session IDs

APP_NAME="google_search_feedback_agent"
USER_ID="user1234"
# SESSION_ID will be generated per interaction

# --- Agent Definition ---
# We need to instruct the agent about the two-phase process.
# Phase 1: Find news, present it, and ask for feedback.
# Phase 2: Using the news and the feedback, summarize.

root_agent = Agent(
    name="feedback_search_agent",
    model="gemini-2.0-flash",
    description="Agent to find news, get feedback, and then summarize.",
    instruction="""You are a helpful AI assistant. Your goal is to provide a summary of news based on user feedback.

    **Phase 1: News Gathering and Feedback Solicitation**
    1. When the user asks for news (e.g., "what's the latest AI news?"), use the Google Search tool to find relevant articles.
    2. Present a list of the titles and very brief snippets (or just links) of the top 3-5 articles you found.
    3. Explicitly ask the user to provide feedback. For example: "Here are some articles I found. Which of these seem most relevant, or are there any specific aspects you'd like me to focus on for the summary?"
    4. **IMPORTANT: Do NOT provide a summary in this first response. Your response should ONLY be the list of articles and the question asking for feedback.**

    **Phase 2: Summarization with Feedback**
    1. After you've presented the articles and asked for feedback, the user will respond.
    2. Use their feedback and the articles you previously found to generate a concise summary.
    3. If the user's feedback is something like "summarize all" or they don't specify, summarize the initially found articles broadly.
    """,
    tools=[google_search]
)

# --- Session and Runner Setup ---
session_service = InMemorySessionService()
# Runner will be created per interaction to ensure a clean session for the multi-turn

# --- Agent Interaction Logic ---
def run_interactive_news_session(initial_query: str):
    """
    Handles the multi-turn interaction for news finding, feedback, and summarization.
    """
    session_id = str(uuid.uuid4()) # Create a unique session ID for this interaction
    print(f"--- Starting New Session: {session_id} ---")

    session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    # Turn 1: Agent finds news and asks for feedback
    print(f"\n[USER] Query: {initial_query}")
    user_content_turn1 = types.Content(role='user', parts=[types.Part(text=initial_query)])
    events_turn1 = runner.run(user_id=USER_ID, session_id=session_id, new_message=user_content_turn1)

    agent_response_turn1_text = ""
    for event in events_turn1:
        if event.is_llm_response() and event.content: # Capture intermediate LLM thought/response
            for part in event.content.parts:
                if part.text:
                    print(f"  [Agent Thinking/Partial]: {part.text.strip()}") # Optional: show agent "thoughts"
        if event.is_tool_request():
            print(f"  [Agent Tool Request]: {event.tool_request.tool_name} with args {event.tool_request.args}")
        if event.is_tool_response():
            print(f"  [Agent Tool Response Received]") # Data can be large, so not printing it fully
        if event.is_final_response():
            agent_response_turn1_text = event.content.parts[0].text
            print(f"\n[AI AGENT - Awaiting Feedback]:\n{agent_response_turn1_text}")
            break # Exit after getting the final response for this turn

    if not agent_response_turn1_text:
        print("[SYSTEM] Agent did not provide a response in Turn 1. Exiting.")
        return

    # Turn 2: Human provides feedback
    human_feedback = input("\n[HUMAN] Your feedback (e.g., 'Focus on the first and third', 'Summarize all', 'Any new chip developments?'): ")
    if not human_feedback.strip():
        print("[SYSTEM] No feedback provided. Agent will proceed with general summary if possible.")
        human_feedback = "Please summarize the articles you found." # Default feedback

    user_content_turn2 = types.Content(role='user', parts=[types.Part(text=human_feedback)])
    events_turn2 = runner.run(user_id=USER_ID, session_id=session_id, new_message=user_content_turn2)

    final_summary_text = ""
    for event in events_turn2:
        if event.is_llm_response() and event.content:
             for part in event.content.parts:
                if part.text:
                    print(f"  [Agent Thinking/Partial for Summary]: {part.text.strip()}")
        if event.is_final_response():
            final_summary_text = event.content.parts[0].text
            print(f"\n[AI AGENT - Final Summary]:\n{final_summary_text}")
            break

    if not final_summary_text:
        print("[SYSTEM] Agent did not provide a final summary. Exiting.")

    print(f"--- Session {session_id} Ended ---")

# --- Run the interaction ---
if __name__ == "__main__":
    run_interactive_news_session("What's the latest AI news regarding open source models?")
    print("\n" + "="*50 + "\n")
    run_interactive_news_session("Any news about climate change solutions this week?")