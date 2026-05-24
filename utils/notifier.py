import os
import requests
import json

def get_discord_webhook_url():
    # 1. Check environment variable
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if url:
        return url
    
    # 2. Check local .env file in root
    try:
        env_paths = [".env", "../.env", "../../.env", "../../../.env"]
        for path in env_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("DISCORD_WEBHOOK_URL="):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                return parts[1].strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error reading .env: {e}")
    
    return None

def send_discord_notification(webhook_url, message):
    if not webhook_url:
        webhook_url = get_discord_webhook_url()
        
    if not webhook_url:
        print("Warning: Discord Webhook URL is not set. Notification skipped.")
        return
        
    payload = {
        "content": message
    }
    try:
        response = requests.post(
            webhook_url, 
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5.0
        )
        if response.status_code == 204:
            print("Discord notification sent successfully.")
        else:
            print(f"Failed to send notification. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending Discord notification: {e}")

if __name__ == "__main__":
    # Test or Manual trigger
    send_discord_notification(None, "🚀 [Manual Test] @z5r10 Training process check.")
