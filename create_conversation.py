import requests

url = "https://tavusapi.com/v2/conversations"

payload = {
    "replica_id": "rb17cf590e15",
    # "conversation_name": "A Meeting with Hassaan",
    # "conversational_context": "You are about to talk to Hassaan, one of the cofounders of Tavus. He loves to talk about AI, startups, and racing cars.",
    "conversation_name": "A Meeting with Karthik",
    "conversational_context": "You are about to talk to Karthik, one of the researchers of Tavus. He loves to talk about AI, startups, and racing cars. Keep your responses short and concise.",
    "custom_greeting": "Hola",
    "properties": {
        "language": "spanish"
    }
    # "properties": {
    #     "max_call_duration": 3600,
    #     "participant_left_timeout": 60,
    #     "participant_absent_timeout": 300,
    #     "enable_recording": True,
    #     "enable_closed_captions": True,
    #     "apply_greenscreen": True,
    #     "language": "spanish",
    #     "recording_s3_bucket_name": "conversation-recordings",
    #     "recording_s3_bucket_region": "us-east-1",
    #     "aws_assume_role_arn": ""
    # }
}
headers = {
    "x-api-key": "94a3b175900048ffbd42d903a83c42ca",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)