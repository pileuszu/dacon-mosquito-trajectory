import requests
import json

def send_discord_notification(webhook_url, message):
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
    URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"
    send_discord_notification(URL, "🚀 [Manual Test] @z5r10 Training process check.")
