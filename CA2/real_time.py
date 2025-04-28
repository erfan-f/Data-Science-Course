from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col, window, to_timestamp, count, lit, udf, avg,desc,sum,to_json, struct
import math
from pyspark.sql.types import DoubleType



transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("customer_id", StringType()),
    StructField("merchant_id", StringType()),
    StructField("merchant_category", StringType()),
    StructField("payment_method", StringType()),
    StructField("amount", IntegerType()),
    StructField("lat", DoubleType()),
    StructField("lng", DoubleType()),
    StructField("device_info", MapType(StringType(), StringType())),
    StructField("status", StringType()),
    StructField("commission_type", StringType()),
    StructField("commission_amount", IntegerType()),
    StructField("vat_amount", IntegerType()),
    StructField("total_amount", IntegerType()),
    StructField("customer_type", StringType()),
    StructField("risk_level", IntegerType()),
    StructField("failure_reason", StringType())
])

#------------------------------------------------------------------------------------------------------

# this part is fine parsa, you just need to get the link and get faimilair with erfan's db
# this part is for connecting mongodb to spark
spark = SparkSession.builder \
    .appName("RealTimeFraudDetection") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/darooghe.customer_history") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/darooghe.customer_history") \
    .getOrCreate()

customer_history_df = spark.read.format("mongo").option("uri", "mongodb://localhost:27017/daroogheDB.transactions").load()

# ---------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
INPUT_TOPIC = 'darooghe.transactions'

valid_transactions_json = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", INPUT_TOPIC) \
    .load()

valid_transactions = valid_transactions_json.select(
    from_json(col("value").cast("string"), transaction_schema).alias("data")
).select("data.*").withColumn(
    "timestamp", 
    to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'")
)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

haversine_udf = udf(haversine, DoubleType())


#this part,parsa
historical_data = spark.read \
    .format("mongo") \
    .option("collection", "customer_history") \
    .load() \
    .select(
        col("customer_id"),
        col("average_transaction_amount"),
        col("last_known_location.lat").alias("hist_lat"),
        col("last_known_location.lng").alias("hist_lng")
    )



def detect_fraud(transactions_df, hist_df):

    velocity_alerts = transactions_df.groupBy(
        col("customer_id"),
        window(col("timestamp"), "2 minutes")
    ).agg(count("*").alias("txn_count")) \
    .filter(col("txn_count") > 5) \
    .select(
        col("customer_id"),
        lit("VELOCITY_CHECK").alias("fraud_type"),
        lit("More than 5 transactions in 2 minutes").alias("description"),
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end")
    )
    
    # this part is completely chatgpt , not tested
    geo_alerts = transactions_df.alias("t1").join(
        transactions_df.alias("t2"),
        (col("t1.customer_id") == col("t2.customer_id")) &
        (col("t1.timestamp") < col("t2.timestamp")) &
        ((col("t2.timestamp").cast("long") - col("t1.timestamp").cast("long")) <= 300)
    ).withColumn(
        "distance_km",
        haversine_udf(col("t1.lat"), col("t1.lng"), col("t2.lat"), col("t2.lng"))
    ).filter(col("distance_km") > 50) \
    .select(
        col("t1.customer_id"),
        lit("GEOGRAPHICAL_IMPOSSIBILITY").alias("fraud_type"),
        lit("Transactions >50km apart within 5 minutes").alias("description"),
        col("t1.timestamp").alias("window_start"),
        col("t2.timestamp").alias("window_end")
    )
    
    # this part might need a few tweaks(not tested)
    amount_alerts = transactions_df.join(
        hist_df, "customer_id"
    ).filter(
        col("amount") > (col("average_transaction_amount") * 10)
    ).select(
        col("customer_id"),
        lit("AMOUNT_ANOMALY").alias("fraud_type"),
        lit("Transaction >10x customer average").alias("description"),
        col("timestamp").alias("window_start"),
        col("timestamp").alias("window_end")
    )
    


    return velocity_alerts.union(geo_alerts).union(amount_alerts)


fraud_alerts = detect_fraud(valid_transactions, historical_data)


query = fraud_alerts.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.fraud_alerts") \
    .option("checkpointLocation", "/tmp/fraud_alerts_checkpoint") \
    .outputMode("update") \
    .start()

query.awaitTermination()


# Real-Time Commission Analytics 

commission_by_type = valid_transactions.groupBy(
    col("commission_type"),
    window(col("timestamp"), "5 minutes")
).agg(
    sum(col("commission_amount")).alias("total_commission")
)

commission_ratio = valid_transactions.groupBy(
    col("merchant_category"),
    window(col("timestamp"), "5 minutes")
).agg(
    (sum("commission_amount") / sum("amount")).alias("commission_ratio")
)


top_merchant = valid_transactions.groupBy(
    col("merchant_id"),
    window(col("timestamp"), "5 minutes")
).agg(
    sum("commission_amount").alias("total_commission")
).orderBy(desc("total_commission")).limit(1)


combined_metrics = commission_by_type.join(
    commission_ratio, ["window"]
).join(
    top_merchant, ["window"]
).select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    col("commission_type"),
    col("total_commission"),
    col("merchant_category"),
    col("commission_ratio"),
    col("merchant_id").alias("highest_commission_merchant"),
    col("total_commission").alias("highest_commission_total")
)


combined_metrics.selectExpr(
    "CAST(null AS STRING) AS key", 
    "to_json(struct(*)) AS value"
).writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commissions_metrics") \
    .option("checkpointLocation", "/tmp/checkpoints/commissions_metrics") \
    .start()
