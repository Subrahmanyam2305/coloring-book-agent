"""
Tavus API integration for Kids Language Learning App
"""
import os
import json
import requests
import time
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv
from threading import Thread

# Load environment variables
load_dotenv()

# API keys and configuration
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY", "94a3b175900048ffbd42d903a83c42ca")  # Default from create_conversation.py
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

def create_tavus_conversation(
    replica_id: str = "rb17cf590e15",  # Default from create_conversation.py
    conversation_name: str = "Language Learning Session",
    character_name: str = None,
    story_context: str = None,
    target_language: str = "spanish",
    custom_greeting: str = "Hello"
) -> Dict[str, Any]:
    """
    Create a new conversation using the Tavus API
    
    Args:
        replica_id: The ID of the Tavus replica to use
        conversation_name: Name for the conversation
        character_name: Name of the character (optional)
        story_context: Story context to use for the conversation
        target_language: Target language for the conversation
        custom_greeting: Custom greeting for the replica
        
    Returns:
        Dict containing conversation details or error
    """
    url = "https://tavusapi.com/v2/conversations"
    
    # Build the conversational context from the story
    context = "You are a friendly language tutor teaching children."
    if character_name:
        context += f" Your name is {character_name}."
    if story_context:
        context += f" You will be talking about this story: {story_context}"
    
    payload = {
        "replica_id": replica_id,
        "conversation_name": conversation_name,
        "conversational_context": context,
        "custom_greeting": custom_greeting,
        "properties": {
            "language": target_language.lower(),
            "enable_closed_captions": True
        }
    }
    
    headers = {
        "x-api-key": TAVUS_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating Tavus conversation: {e}")
        return {"error": str(e)}

def get_conversation_url(conversation_id: str) -> str:
    """
    Get the URL for a Tavus conversation
    
    Args:
        conversation_id: The ID of the conversation
        
    Returns:
        URL string for the conversation
    """
    # According to the API response, the format seems to be https://tavus.daily.co/CONVERSATION_ID
    return f"https://tavus.daily.co/{conversation_id}"

def format_story_for_tavus(story_scenes: List[str]) -> str:
    """
    Format multiple story scenes into a single context for Tavus
    
    Args:
        story_scenes: List of story scenes
        
    Returns:
        Formatted story context
    """
    if not story_scenes:
        return ""
    
    formatted_story = "Story Context: "
    for i, scene in enumerate(story_scenes):
        formatted_story += f"\nScene {i+1}: {scene}"
        
    return formatted_story

def get_default_story(character_name: str, child_name: str, target_language: str) -> List[str]:
    """
    Generate a default story when none is provided
    
    Args:
        character_name: Name of the character in the story
        child_name: Name of the child/student
        target_language: Target language for the conversation
        
    Returns:
        List of story scenes
    """
    # Create a simple default story with the character and child
    return [
        f"Meet {character_name}, a friendly character who loves to teach {target_language}.",
        f"{character_name} and {child_name} become friends and learn new words together.",
        f"They practice counting, colors, and simple phrases in {target_language}.",
        f"{character_name} shows {child_name} pictures of objects and teaches their names.",
        f"At the end of the day, {child_name} can say several new words in {target_language}!"
    ]

def initialize_tavus_conversation_from_story(
    story_scenes: List[str] = None,
    character_name: str = "Friendly Character",
    target_language: str = "English",
    child_name: str = "Student"
) -> Dict[str, Any]:
    """
    Initialize a Tavus conversation based on a generated story
    
    Args:
        story_scenes: List of story scenes (optional)
        character_name: Name of the character in the story
        target_language: Target language for the conversation
        child_name: Name of the child/student
        
    Returns:
        Dict containing conversation details or error
    """
    # If no story is provided, use the default story
    if not story_scenes:
        story_scenes = get_default_story(character_name, child_name, target_language)
        print(f"Using default story: {story_scenes}")
        
    # Format story for Tavus context
    story_context = format_story_for_tavus(story_scenes)
    
    # Create conversation name
    conversation_name = f"{child_name}'s {target_language} Learning Session with {character_name}"
    
    # Create custom greeting based on target language
    greetings = {
        "spanish": "¡Hola!",
        "french": "Bonjour!",
        "english": "Hello!",
    }
    
    custom_greeting = greetings.get(target_language.lower(), "Hello!")
    
    # Create the conversation
    result = create_tavus_conversation(
        conversation_name=conversation_name,
        character_name=character_name,
        story_context=story_context,
        target_language=target_language,
        custom_greeting=custom_greeting
    )
    
    return result

# Function to handle conversation joining in background thread
def join_conversation_room(conversation_id: str, conversation_url: str) -> None:
    """
    Join a Tavus conversation room using the interaction_protocol_tavus.py script
    
    Args:
        conversation_id: The ID of the conversation
        conversation_url: URL of the conversation
        
    Note:
        This function runs in a background thread to avoid blocking the main app
    """
    import subprocess
    import os.path
    
    # Check if both conversation_id and conversation_url are valid
    if not conversation_id or not conversation_url:
        print(f"Error: Invalid conversation details - ID: {conversation_id}, URL: {conversation_url}")
        return
    
    # Check if the script exists
    script_path = "/Users/subrahmanyam.arunachalam/Documents/personal/dumb-hack/coloring-book-agent/interaction_protocol_tavus.py"
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return
        
    try:
        print(f"Joining conversation: ID={conversation_id}, URL={conversation_url}")
        
        # Call the interaction_protocol_tavus.py script with conversation details
        # The conversation URL should be sufficient as the conversation ID is included in it
        cmd = [
            "python", 
            script_path,
            "--conversation_url", conversation_url
        ]
        
        # Start the process in the background with output redirection
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print the initial output (non-blocking)
        print(f"Started conversation process with PID: {process.pid}")
        
    except Exception as e:
        print(f"Error joining conversation room: {e}")
