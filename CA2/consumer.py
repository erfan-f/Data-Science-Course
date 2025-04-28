import json
import datetime
import os
from confluent_kafka import Consumer, Producer

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
INPUT_TOPIC = 'darooghe.transactions'
ERROR_TOPIC = 'darooghe.error_logs'
VALIDATED_OUTPUT_FILE = 'content/validated_transactions.json'
VALID_OS = ["iOS", "Android"]

os.makedirs('content', exist_ok=True)

consumer = Consumer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': 'txn-validator',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe([INPUT_TOPIC])

producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

def convert_and_validate(msg: dict):
    try:
        required_keys = [
            "transaction_id", "timestamp", "customer_id", "merchant_id",
            "merchant_category", "payment_method", "amount", "location",
            "device_info", "status", "commission_type", "commission_amount",
            "vat_amount", "total_amount", "customer_type", "risk_level", "failure_reason"
        ]

        for key in required_keys:
            if key not in msg:
                raise ValueError(f"Missing key: {key}")
        
        if "lat" not in msg["location"] or "lng" not in msg["location"]:
            raise ValueError("Missing lat/long in location")

        filtered_msg = {
            "transaction_id": str(msg['transaction_id']),
            "timestamp": str(msg['timestamp']),
            "customer_id": str(msg['customer_id']),
            "merchant_id": str(msg['merchant_id']),
            "merchant_category": str(msg['merchant_category']),
            "payment_method": str(msg['payment_method']),
            "amount": int(msg['amount']),
            "lat": float(msg["location"]["lat"]),
            "lng": float(msg["location"]["lng"]),
            "device_info": msg.get("device_info", {}),
            "status": str(msg['status']),
            "commission_type": str(msg['commission_type']),
            "commission_amount": int(msg['commission_amount']),
            "vat_amount": int(msg['vat_amount']),
            "total_amount": int(msg['total_amount']),
            "customer_type": str(msg['customer_type']),
            "risk_level": int(msg['risk_level']),
            "failure_reason": str(msg['failure_reason']),
        }
        return filtered_msg
    
    except Exception as e:
        print(f"[❌ Conversion error]: {e}")
        return None

def validate_transaction(txn, kafka_time):
    errors = []
    
    if txn["total_amount"] != txn["amount"] + txn["vat_amount"] + txn["commission_amount"]:
        errors.append("ERR_AMOUNT")

    txn_time = datetime.datetime.fromisoformat(txn["timestamp"].replace("Z", ""))
    if txn_time > kafka_time or txn_time < kafka_time - datetime.timedelta(days=1):
        errors.append("ERR_TIME")

    if txn["payment_method"] == "mobile":
        os_type = txn.get("device_info", {}).get("os")
        if os_type not in VALID_OS:
            errors.append("ERR_DEVICE")
   
    return errors

print("🔄 Starting Kafka consumer...")
try:
    with open(VALIDATED_OUTPUT_FILE, 'a') as outfile:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"⚠️ Error: {msg.error()}")
                continue

            try:
                raw_txn = json.loads(msg.value())
                converted_txn = convert_and_validate(raw_txn)

                if converted_txn is None:
                    continue

                kafka_time = datetime.datetime.utcnow()
                errors = validate_transaction(converted_txn, kafka_time)

                if errors:
                    for err in errors:
                        error_event = {
                            "transaction_id": converted_txn.get("transaction_id"),
                            "error_code": err,
                            "timestamp": kafka_time.isoformat() + "Z"
                        }
                        producer.produce(ERROR_TOPIC, key=converted_txn.get("transaction_id"), value=json.dumps(error_event))
                        print(f"🚫 Sent error to {ERROR_TOPIC}: {error_event}")
                else:
                    outfile.write(json.dumps(converted_txn) + "\n")
                    print(f"✅ Valid transaction: {converted_txn['transaction_id']}")

            except Exception as e:
                print(f"[❌ Deserialization or processing error]: {e}")

            producer.flush()

except KeyboardInterrupt:
    print("🛑 Shutting down consumer...")

finally:
    consumer.close()
