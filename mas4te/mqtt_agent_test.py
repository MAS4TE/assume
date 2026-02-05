
import json
import time
import threading
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

BROKER = "localhost"  # or the IP of your MQTT broker
PORT = 1883

TOPIC_MARKET_STATUS = "mas4te/market/status"
TOPIC_BIDS_AGENT1 = "mas4te/bids/agent1"
TOPIC_RESULTS_AGENT1 = "mas4te/results/agent1"

def on_connect(client, userdata, flags, rc, properties):
    print("Connected to MQTT broker with result code:", rc)
    client.subscribe(TOPIC_BIDS_AGENT1)

    # Publish market open
    message = {"status": "market_open"}
    client.publish(TOPIC_MARKET_STATUS, json.dumps(message), retain=True)
    print("Market status published")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")

client = mqtt.Client(client_id="test_agent", callback_api_version=CallbackAPIVersion.V2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

# Run the MQTT client in a separate thread
thread = threading.Thread(target=client.loop_forever)
thread.daemon = True
thread.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting")
