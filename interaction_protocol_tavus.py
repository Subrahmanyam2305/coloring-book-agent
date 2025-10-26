import argparse
import json
import os
import time
import wave
from enum import Enum
from typing import Any, Mapping, Optional
import requests
from dotenv import load_dotenv
from daily import CallClient, Daily, EventHandler, VirtualMicrophoneDevice
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
from flask import Flask, Response, stream_with_context, request, jsonify
from flask_cors import CORS
from queue import Queue
from threading import Thread
from anthropic import Anthropic
import sys

# # model_name = "deepseek-ai/deepseek-math-7b-instruct"
# model_name = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/deepseek-checkpoints/deepseek-math-7b-rl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

load_dotenv()

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
ENV_TO_TEST = os.getenv("ENV_TO_TEST", "prod")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Add your Anthropic API key to .env file

# Initialize Anthropic client
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

assistant_utterance = None
gpu_joined = False
assistant_utterance_time = None
warm_boot_time = None
questions_seen = {}
# global_lock = False

# --- Added for SSE ---
utterance_queue = Queue()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add new global variables for current conversation settings and client
current_conversation_settings = {
    'source_lang': 'english',
    'target_lang': 'spanish',
    'topic': 'restaurant'
}

# Add topic descriptions as a global constant
TOPIC_DESCRIPTIONS = {
    'restaurant': 'Ordering food at a restaurant',
    'travel': 'Booking a hotel room',
    'shopping': 'Shopping for clothes'
}

# Global variables for Daily client and conversation info
client_global = None
conversation_id_global = None
conversation_url_global = None

def get_topic_description():
    return TOPIC_DESCRIPTIONS.get(current_conversation_settings['topic'], TOPIC_DESCRIPTIONS['restaurant'])

def get_claude_response_translation_alone(utterance: str, source_lang: str, target_lang: str) -> str:
    """Get a response from Claude for the given utterance."""
    client = Anthropic()

    try:
        message = client.messages.create(
            # model="claude-3-opus-20240229",
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.7,
            system=f"Act as a language tutor whose job is to translate from {source_lang} to {target_lang}. " \
                   f"Keep your responses short and concise. Don't make it more than 2 sentences." \
                   f"You must use the tool 'generate_response' to generate the translation. Whenever you are responding dont use any thinking or reasoning, just respond with the tool call.",
            messages=[
                {"role": "user", "content": utterance}
            ],
            tools=[{
                "name": "generate_response",
                "description": f"Translate the {source_lang} message to {target_lang}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_translation": {
                            "type": "string",
                            "description": f"The translation of the message in {target_lang}"
                        }
                    },
                    "required": ["target_translation"]
                }
            }]
        )
        
        print(f"DEBUG: Claude response translation alone: {message}")
        # Check if Claude used the tool
        if message.stop_reason == "tool_use":
            for content in message.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    if tool_name == "generate_response":
                        # Extract the translation
                        translation = tool_input["target_translation"]
                        return {"English": translation}  # Keep English key for backwards compatibility
        else:
            # Handle the case where Claude didn't use the tool
            response_text = message.content[0].text
            if response_text is not None and response_text != "":
                return {"English": response_text.strip()}
            else:
                return {"English": "Translation not available"}

    except Exception as e:
        print(f"Error getting Claude response: {e}")
        return None

def get_claude_response(utterance: str, topic: str, source_lang: str, target_lang: str) -> str:
    """Get a response from Claude for the given utterance."""
    client = Anthropic()

    try:
        message = client.messages.create(
            # model="claude-3-opus-20240229",
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.7,
            system=f"Act as a {source_lang} language tutor. Engage in a conversation with me, correcting my grammar and vocabulary as needed. " \
                   f"Respond to my messages in {source_lang}, and provide {target_lang} translations when necessary to help me understand. Our conversation topic is {topic}. " \
                   f"Keep your responses short and concise. Don't make it more than 2 sentences." \
                   f"You must use the tool 'generate_response' to generate a response in {source_lang} with a {target_lang} translation. Whenever you are responding dont use any thinking or reasoning, just respond with the tool call.",
            messages=[
                {"role": "user", "content": utterance}
            ],
            tools=[{
                "name": "generate_response",
                "description": f"Generate a response in {source_lang} with a {target_lang} translation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_response": {
                            "type": "string",
                            "description": f"The response in {source_lang}"
                        },
                        "target_translation": {
                            "type": "string",
                            "description": f"The {target_lang} translation of the response"
                        }
                    },
                    "required": ["source_response", "target_translation"]
                }
            }]
        )
        
        # print(f"DEBUG: Claude response: {message}")
        # Check if Claude used the tool
        if message.stop_reason == "tool_use":
            for content in message.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    if tool_name == "generate_response":
                        # Extract the responses
                        source_response = tool_input["source_response"]
                        target_translation = tool_input["target_translation"]
                        return {
                            "Spanish": source_response,  # Keep Spanish key for backwards compatibility
                            "English": target_translation  # Keep English key for backwards compatibility
                        }
        else:
            # Handle the case where Claude didn't use the tool
            response_text = message.content[0].text
            parts = response_text.split('\n\n')
            if len(parts) >= 2:
                return {"Spanish": parts[0].strip(), "English": parts[1].strip()}
            return {"Spanish": response_text, "English": "Translation not available"}

    except Exception as e:
        print(f"Error getting Claude response: {e}")
        return None

@app.route('/update-conversation', methods=['POST'])
def update_conversation():
    """Endpoint to handle conversation updates from frontend."""
    try:
        data = request.get_json()
        new_url = data.get('conversation_url')
        
        # Update current conversation settings
        global current_conversation_settings
        current_conversation_settings.update({
            'source_lang': data.get('source_lang', current_conversation_settings['source_lang']),
            'target_lang': data.get('target_lang', current_conversation_settings['target_lang']),
            'topic': data.get('topic', current_conversation_settings['topic'])
        })
        # print(f"DEBUG: Current conversation settings: {current_conversation_settings}")
        if not new_url:
            return jsonify({'error': 'No conversation URL provided'}), 400
            
        success = handle_conversation_change(new_url)
        
        if success:
            return jsonify({
                'status': 'success', 
                'message': 'Conversation updated',
                'settings': current_conversation_settings
            })
        else:
            return jsonify({'error': 'Failed to update conversation'}), 500
            
    except Exception as e:
        print(f"Error in update_conversation endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def event_stream():
    while True:
        try:
            # Check queue size instead of length
            if not utterance_queue.empty():
                utterance = utterance_queue.get_nowait()  # Non-blocking get
                print(f"DEBUG: Got utterance from queue: {utterance}")
                
                if utterance is None or utterance == "":  # Allow graceful shutdown if needed
                    print("DEBUG: Received None utterance, breaking stream")
                    break
                
                # Use current settings for translation
                utterance_translation = get_claude_response_translation_alone(
                    utterance, 
                    current_conversation_settings['source_lang'],
                    current_conversation_settings['target_lang']
                )
                
                # First, send the user's utterance
                user_message = {
                    "type": "user_utterance",
                    "text": utterance,
                    "spanish_text": utterance,  # This will be the original text
                    "english_text": utterance_translation["English"] if utterance_translation else utterance,
                    "source_lang": current_conversation_settings['source_lang'],
                    "target_lang": current_conversation_settings['target_lang']
                }
                data = f"data: {json.dumps(user_message)}\n\n"
                yield data
                # Force flush after each event
                if hasattr(sys.stdout, 'flush'):
                    sys.stdout.flush()

                # Get Claude's response using current settings
                current_topic_description = get_topic_description()  # Get current topic description
                claude_response = get_claude_response(
                    utterance, 
                    current_topic_description,
                    current_conversation_settings['source_lang'],
                    current_conversation_settings['target_lang']
                )
                # print(f"DEBUG: Raw Claude response: {claude_response}")
                
                if claude_response is not None:
                    if isinstance(claude_response, dict) and "Spanish" in claude_response and "English" in claude_response:
                        # Handle structured response from tool call
                        ai_message = {
                            "type": "ai_response",
                            "text": claude_response["Spanish"],  # For backwards compatibility
                            "spanish_text": claude_response["Spanish"],
                            "english_text": claude_response["English"],
                            "source_lang": current_conversation_settings['source_lang'],
                            "target_lang": current_conversation_settings['target_lang']
                        }
                    else:
                        # Handle plain text response
                        ai_message = {
                            "type": "ai_response",
                            "text": str(claude_response),
                            "spanish_text": str(claude_response),
                            "english_text": "Translation not available",
                            "source_lang": current_conversation_settings['source_lang'],
                            "target_lang": current_conversation_settings['target_lang']
                        }
                    
                    # print(f"DEBUG: Formatted AI message: {ai_message}")
                    data = f"data: {json.dumps(ai_message)}\n\n"
                    yield data
                    # Force flush after each event
                    if hasattr(sys.stdout, 'flush'):
                        sys.stdout.flush()
                    print("DEBUG: AI response yielded successfully")
                else:
                    print("DEBUG: Claude response is None, skipping")
                
                utterance_queue.task_done()
            else:
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)  # Wait between checks
        except Exception as e:
            print(f"Error in event stream: {e}")
            # Don't break the stream on error, continue processing
            time.sleep(0.1)
            continue

@app.route('/listen-utterances')
def listen_utterances():
    # Get conversation settings from query parameters and update global settings
    # print(f"DEBUG: Received request with query parameters: {request.args}")
    global current_conversation_settings
    current_conversation_settings.update({
        'topic': request.args.get('topic', current_conversation_settings['topic']),
        'source_lang': request.args.get('source_lang', current_conversation_settings['source_lang']),
        'target_lang': request.args.get('target_lang', current_conversation_settings['target_lang'])
    })

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Content-Encoding': 'none',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Transfer-Encoding': 'chunked'
        }
    )

class TestType(Enum):
    FULL = "full"
    # ECHO = "echo"


class RoomHandler(EventHandler):
    def __init__(self):
        super().__init__()

    def on_app_message(self, message, sender: str) -> None:
        global client_global, conversation_id_global, conversation_url_global
        try:
            if isinstance(message, str):
                json_message = json.loads(message)
            else:
                json_message = message
        except Exception as e:
            print(f"Error parsing message: {e}")
            return

        if json_message["event_type"] == "conversation.utterance":
            utterance_text = json_message['properties']['speech']
            role_text = json_message["properties"]["role"]
            # print(f"DEBUG: Received utterance - Text: {utterance_text}, Role: {role_text}")
            
            if role_text == "replica":  # Changed to capture non-replica speech
                # print(f"DEBUG: Queueing utterance from {role_text}")
                utterance_queue.put(utterance_text)
                print(f"DEBUG: Successfully queued utterance")
            else:
                print(f"DEBUG: Skipping replica utterance")
        elif json_message["event_type"] == "system.replica_joined":
            global gpu_joined, warm_boot_time
            gpu_joined = True
            warm_boot_time = time.time()

# def call_deepseek_llm(question):
#     parse_question = question.split("=")[0].strip()
#     messages = [
#         {"role": "user", "content": f"what is {parse_question}, solve this problem and put your final answer within " + "\\boxed{}"}
#     ]
#     input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
#     outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

#     result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
#     cleaned_result = clean_math_text(result)
#     print(cleaned_result)
#     return cleaned_result


def call_joined(join_data: Optional[Mapping[str, Any]], client_error: Optional[str]):
    if client_error:
        raise RuntimeError(f"Bot failed to join call: {client_error}, {join_data}")
    else:
        print(f"call_joined ran successfully")


def join_room(call_client: CallClient, url: str, conversation_id: str):
    try:
        call_client.join(
            meeting_url=url,
            meeting_token=get_meeting_token(
                conversation_id, DAILY_API_KEY, True, None, False
            ),
            client_settings={
                "inputs": {
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": "microphone"},
                    }
                },
                "publishing": {
                    "microphone": {
                        "isPublishing": True,
                        "sendSettings": {"channelConfig": "mono", "bitrate": 16000},
                    }
                },
            },
            completion=call_joined,
        )
        print(f"Joined room: {url}")
    except Exception as e:
        print(f"Error joining room: {e}")
        raise


def send_text_echo(
    call_client: CallClient,
    conversation_id: str,
    text: str = "This is the text the replica will speak.",
):
    call_client.send_app_message(
        {
            "message_type": "conversation",
            "event_type": "conversation.echo",
            "conversation_id": conversation_id,
            "properties": {"text": text},
        }
    )


def get_meeting_token(
    room_name: str,
    daily_api_key: str,
    has_presence: bool,
    token_expiry: Optional[float],
    enable_transcription: bool,
) -> str:
    assert daily_api_key, "Must provide `DAILY_API_KEY` env var"

    if not token_expiry:
        token_expiry = time.time() + 600
    res = requests.post(
        "https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "room_name": room_name,
                "is_owner": True,
                "enable_live_captions_ui": enable_transcription,
                "exp": token_expiry,
                "permissions": {"hasPresence": has_presence},
            }
        },
    )
    assert res.status_code == 200, f"Unable to create meeting token: {res.text}"

    meeting_token = res.json()["token"]
    return str(meeting_token)


def init_daily():
    Daily.init()
    output_handler = RoomHandler()
    client = CallClient(event_handler=output_handler)
    vmic = Daily.create_microphone_device("microphone", non_blocking=True, channels=1)
    return client, vmic


def init_daily_client():
    Daily.init()
    output_handler = RoomHandler()
    client = CallClient(event_handler=output_handler)
    return client


# def run_heartbeat(conversation_url: str, conversation_id: str):
#     global client_global, conversation_id_global, conversation_url_global
#     client_global = init_daily_client()
#     # Join the room
#     conversation_id_global = conversation_id
#     conversation_url_global = conversation_url
#     join_room(client_global, conversation_url_global, conversation_id_global)
#     while True:
#         time.sleep(1)


def handle_conversation_change(new_url: str) -> bool:
    """
    Handle switching to a new conversation room.
    Args:
        new_url: The URL of the new conversation room
    Returns:
        bool: True if switch was successful, False otherwise
    """
    global client_global, conversation_id_global, conversation_url_global
    
    try:
        # Extract conversation ID from URL
        conversation_id = new_url.split('/')[-1]
        
        # If we already have a client and are in a room, leave it
        if client_global and conversation_url_global:
            try:
                client_global.leave()
                print(f"Left room: {conversation_url_global}")
            except Exception as e:
                print(f"Error leaving room: {e}")
        
        # Initialize new client if needed
        if not client_global:
            client_global = init_daily_client()
            print("Initialized new Daily client")
        
        # Update globals
        conversation_id_global = conversation_id
        conversation_url_global = new_url
        
        # Join new room
        join_room(client_global, new_url, conversation_id)
        print(f"Successfully joined new room: {new_url}")
        
        return True
        
    except Exception as e:
        print(f"Error in handle_conversation_change: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversation_id", type=str, help="Initial conversation ID (optional)"
    )
    parser.add_argument(
        "--conversation_url", type=str, help="Initial conversation URL (optional)"
    )
    args = parser.parse_args()

    # --- Added for SSE ---
    # Start Flask server in a background thread
    # Use port 5001 as configured in index.html
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    print("Flask SSE server started on port 5001")
    # --- End SSE Additions ---

    if args.conversation_url and args.conversation_id:
        # If initial conversation details provided, join that room
        success = handle_conversation_change(args.conversation_url)
        if not success:
            print("Failed to join initial conversation room")
            return
    else:
        print("Waiting for first conversation to be created via frontend...")
    
    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()