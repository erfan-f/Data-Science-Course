from confluent_kafka import Consumer, KafkaException, KafkaError
import json

# Kafka consumer configuration
consumer_config = {
    'bootstrap.servers': 'localhost:9092',  # Replace with your Kafka bootstrap servers
    'group.id': 'fraud-alerts-consumer-group',
    'auto.offset.reset': 'earliest',  # Start reading from the beginning of the topic
}

# Create Kafka consumer
consumer = Consumer(consumer_config)

# Subscribe to the 'darooghe.fraud_alerts' topic
consumer.subscribe(['darooghe.fraud_alerts'])

# Function to process the fraud alert message
def process_fraud_alert(message):
    try:
        # Deserialize the JSON message
        alert_data = json.loads(message.value().decode('utf-8'))
        print(f"Received fraud alert: {alert_data}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Consume messages indefinitely
try:
    while True:
        msg = consumer.poll(timeout=1.0)  # Poll for new messages
        
        if msg is None:
            continue  # No message, continue polling
        
        # if msg.error():
        #     if msg.error().code() == KafkaError._PARTITION_EOF:
        #         print(f"End of partition reached {msg.partition} at offset {msg.offset()}")
        #     else:
        #         raise KafkaException(msg.error())
        # else:
        #     # Process the valid fraud alert message
        process_fraud_alert(msg)

except KeyboardInterrupt:
    print("Consumer stopped")

finally:
    # Close the consumer gracefully
    consumer.close()
