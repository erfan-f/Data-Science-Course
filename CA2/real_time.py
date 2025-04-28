from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col, window, to_timestamp, count, lit, udf, avg, desc, sum, to_json, struct, unix_timestamp, rank
from pyspark.sql.functions import window, count, sum as sum_, desc
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import math


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

# -------------------------------------------------Spark Streaming Application------------------------------------------------

spark = SparkSession.builder \
    .appName("RealTimeFraudDetection") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,"
            "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/daroogheDB.transactions") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/daroogheDB.transactions") \
    .getOrCreate()

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
INPUT_TOPIC = 'darooghe.transactions'
NEW_INSIGHT = 'darooghe.analytics'

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

# Example Insight: Number of transactions per customer in 1-minute windows sliding every 20 seconds
transactions_per_customer = valid_transactions \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        window(col("timestamp"), "1 minute", "20 seconds"),
        col("customer_id")
    ) \
    .agg(count("*").alias("transaction_count")) \
    .select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "customer_id",
        "transaction_count"
    )

# Write insight to a new Kafka topic
query = transactions_per_customer \
    .selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("topic", NEW_INSIGHT) \
    .option("checkpointLocation", "/tmp/spark-checkpoint-analytics") \
    .outputMode("update") \
    .start()

query.awaitTermination()

# ---------------------------------------------------Fraud Detection System----------------------------------------------------

FRAUD_TOPIC = 'darooghe.fraud_alerts'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

haversine_udf = udf(haversine, DoubleType())

def detect_fraud(transactions_df, customer_avg_df):
    # Velocity Check
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

    # Geographical Impossibility
    geo_alerts = transactions_df.alias("t1").join(
        transactions_df.alias("t2"),
        (col("t1.customer_id") == col("t2.customer_id")) &
        (unix_timestamp(col("t2.timestamp")) - unix_timestamp(col("t1.timestamp")) <= 300) &
        (unix_timestamp(col("t2.timestamp")) > unix_timestamp(col("t1.timestamp")))
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

    # Amount Anomaly
    amount_alerts = transactions_df.join(
        customer_avg_df, "customer_id"
    ).filter(
        col("amount") > (col("average_amount") * 10)
    ).select(
        col("customer_id"),
        lit("AMOUNT_ANOMALY").alias("fraud_type"),
        lit("Transaction amount >10x customer average").alias("description"),
        col("timestamp").alias("window_start"),
        col("timestamp").alias("window_end")
    )

    return velocity_alerts.union(geo_alerts).union(amount_alerts)


customer_history_df = spark.read.format("mongo").load()
customer_avg_df = customer_history_df.groupBy("customer_id").agg(
    avg("amount").alias("average_amount")
)

fraud_alerts = detect_fraud(valid_transactions, customer_avg_df)

fraud_alerts_query = fraud_alerts.selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("topic", FRAUD_TOPIC) \
    .option("checkpointLocation", "/tmp/fraud_alerts_checkpoint") \
    .outputMode("append") \
    .start()

fraud_alerts_query.awaitTermination()

# -------------------------------------------------Real-Time Commission Analytics --------------------------------------------------

# 1. Total commission by type per 1 minute
commission_by_type = valid_transactions.groupBy(
    window(col("timestamp"), "1 minute"),
    col("commission_type")
).agg(
    sum(col("commission_amount")).alias("total_commission")
)

commission_by_type_query = commission_by_type.selectExpr(
    "to_json(struct(*)) AS value"
).writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_by_type") \
    .option("checkpointLocation", "/tmp/checkpoints/commission_by_type") \
    .outputMode("update") \
    .start()


# 2. Commission ratio by merchant category per 5 minutes
commission_ratio = valid_transactions.groupBy(
    window(col("timestamp"), "5 minutes"),
    col("merchant_category")
).agg(
    (sum("commission_amount") / sum("amount")).alias("commission_ratio")
)

commission_ratio_query = commission_ratio.selectExpr(
    "to_json(struct(*)) AS value"
).writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.commission_ratio") \
    .option("checkpointLocation", "/tmp/checkpoints/commission_ratio") \
    .outputMode("update") \
    .start()


# 3. Highest commission-generating merchants per 5 minutes
merchant_commission = valid_transactions.groupBy(
    window(col("timestamp"), "5 minutes"),
    col("merchant_id")
).agg(
    sum("commission_amount").alias("total_commission")
)

# Use ranking to get top merchant
w = Window.partitionBy("window").orderBy(desc("total_commission"))

top_merchants = merchant_commission.withColumn(
    "rank", rank().over(w)
).filter(
    col("rank") == 1
).drop("rank")

top_merchants_query = top_merchants.selectExpr(
    "to_json(struct(*)) AS value"
).writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "darooghe.top_merchant") \
    .option("checkpointLocation", "/tmp/checkpoints/top_merchant") \
    .outputMode("complete") \
    .start()

commission_by_type_query.awaitTermination()
commission_ratio_query.awaitTermination()
top_merchants_query.awaitTermination()
