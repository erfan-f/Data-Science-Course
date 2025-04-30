from darooghe_pulse import *

def read_trasnaction():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'transaction-validator-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['darooghe.transactions'])
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            
            try:
                transaction = json.loads(msg.value().decode('utf-8'))
                print(f"Validating order: {transaction['transaction_id']}")
                
                validated_transaction = validate_transaction(transaction)
                
                print(f"Transaction {transaction['transaction_id']} validated. Status: Accepted")
            except Exception as error:
                error_message = {
                    "transaction_id": transaction.get("transaction_id", "UNKNOWN"),
                    "error_code": str(error),
                    "transaction": transaction
                }
                producer.produce(
                    topic='darooghe.error_logs',
                    value=json.dumps(error_message).encode('utf-8')
                )
                producer.poll(0)
                # print(f'Error processing order: {transaction.get("transaction_id", "UNKNOWN")}')
                print(f'Error processing order: {str(error)}')
    except KeyboardInterrupt:
        print("reading order has been stopped sucessfully")
    finally:
        consumer.close()
        producer.flush()





def validate_transaction(transaction):
    expected_total = transaction["amount"] + transaction["vat_amount"] + transaction["commission_amount"]
    actual_total = transaction["total_amount"]

    if abs(expected_total - actual_total) > 5: 
        print(f"Checking total: {transaction['total_amount']} vs {transaction['amount']} + {transaction['vat_amount']} + {transaction['commission_amount']} ={ transaction['amount'] + transaction['vat_amount'] + transaction['commission_amount']} ")
        raise ValueError("ERR_AMOUNT")


    tx_time = datetime.datetime.fromisoformat(transaction["timestamp"].replace("Z", ""))
    now = datetime.datetime.utcnow()
    if tx_time > now or (now - tx_time).total_seconds() > 86400:
        raise ValueError("ERR_TIME")


    if transaction["payment_method"] == "mobile":
        if transaction.get("device_info", {}).get("os") not in ["iOS", "Android"]:
            raise ValueError("ERR_DEVICE")

    return transaction


read_trasnaction()
