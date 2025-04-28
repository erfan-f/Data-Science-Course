# window size:
df_with_watermark = df.withWatermark("timestamp_column", "2 minutes")

windowed_df = df_with_watermark.groupBy(
    window("timestamp_column", "1 minute", "20 seconds")
).agg(
    functions.count("*").alias("transaction_count"),
    functions.sum("amount").alias("total_amount")
)



# Enable Checkpointing
query = result.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "processed_topic") \
    .option("checkpointLocation", "/tmp/spark_checkpoints") \
    .start()


