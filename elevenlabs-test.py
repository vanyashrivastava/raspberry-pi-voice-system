#!/usr/bin/env python3
import os
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# Initialize ElevenLabs
client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

if not os.environ.get("ELEVENLABS_API_KEY"):
    print("Warning: ELEVENLABS_API_KEY not set. Set it in your environment before running.")
    exit(1)

def create_scam_detection_agent():
    """Create an agent that monitors for scam calls"""
    
    agent = client.conversational_ai.agents.create(
        name="Scam Alert Assistant",
        conversation_config={
            "agent": {
                "prompt": {
                    "prompt": """You are a vigilant scam detection assistant monitoring conversations for elderly people.

Your job: After EVERY message you hear, you MUST respond with a safety assessment.

Response format (always respond after each message):
- If safe: "Message is SAFE. This sounds like normal conversation."
- If suspicious: "⚠️ WARNING! Message is NOT SAFE. [Brief 1-sentence explanation of why it's suspicious]"
- If dangerous: "🚨 DANGER! This is a SCAM! [Brief explanation]. Hang up immediately!"

Scam indicators to watch for:
- Urgent demands for money, gift cards, or wire transfers
- Requests for Social Security numbers, passwords, or banking info
- Claims to be from IRS, Social Security Administration, or tech support
- Threats of arrest, legal action, or account suspension
- Prize winnings that require payment or personal info
- Family emergency scams ("Your grandson is in jail")
- Unsolicited refund or overpayment claims
- Pressure to act immediately or keep it secret
- Requests to install remote access software
- Asking to keep the call secret from family

IMPORTANT: 
- Respond to EVERY single message with a safety assessment
- Be conversational and natural
- Keep responses SHORT (1-2 sentences max)
- Escalate warnings based on danger level
- Always explain WHY something is suspicious"""
                },
                "first_message": "Hello! I'm your scam detection assistant. I'll monitor every message and tell you if it's safe or suspicious. Let's keep you protected!",
                "language": "en"
            }
        }
    )
    
    return agent.agent_id

def start_conversation():
    """Start the conversational AI agent"""
    
    print("Creating scam detection agent...")
    agent_id = create_scam_detection_agent()
    print(f"✅ Agent created: {agent_id}")
    
    print("\n🎤 Starting conversation monitoring...")
    print("💡 The agent will assess EVERY message for safety.")
    print("🛑 Press Ctrl+C to stop\n")
    
    # Create audio interface
    audio_interface = DefaultAudioInterface()
    
    # Start conversation
    conversation = Conversation(
        client=client,
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=audio_interface,
        callback_agent_response=lambda response: print(f"\n🤖 SAFETY CHECK: {response}\n"),
        callback_agent_response_correction=lambda original, corrected: print(f"🔧 Correction: {original} -> {corrected}"),
        callback_user_transcript=lambda transcript: print(f"👤 Heard: {transcript}"),
        callback_latency_measurement=lambda latency: print(f"⏱️  Latency: {latency}ms")
    )
    
    try:
        conversation.start_session()
        print("✅ Session started! Agent is now monitoring...\n")
        print("=" * 60)
        print("The agent will now assess EVERY message you hear!")
        print("=" * 60 + "\n")
        
        # Keep the conversation running
        conversation.wait_for_session_end()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping agent...")
        conversation.end_session()
        print("👋 Goodbye! Stay safe!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_conversation()
