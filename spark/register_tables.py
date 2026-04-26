import os
import sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
os.environ["PYSPARK_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("RegisterTables")
         .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
         .config("spark.driver.extraJavaOptions",
                 "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                 "--add-opens=java.base/java.nio=ALL-UNNAMED "
                 "--add-opens=java.base/java.lang=ALL-UNNAMED "
                 "--add-opens=java.base/java.util=ALL-UNNAMED")
         .config("spark.executor.extraJavaOptions",
                 "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                 "--add-opens=java.base/java.nio=ALL-UNNAMED "
                 "--add-opens=java.base/java.lang=ALL-UNNAMED "
                 "--add-opens=java.base/java.util=ALL-UNNAMED")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

spark.sql("""
    CREATE TABLE IF NOT EXISTS traffic_features
    USING parquet
    LOCATION 'hdfs://localhost:9000/traffic/features/'
""")

spark.sql("""
    CREATE TABLE IF NOT EXISTS traffic_processed
    USING parquet
    LOCATION 'hdfs://localhost:9000/traffic/processed/'
""")

print("Tables registered:")
spark.sql("SHOW TABLES").show()

print("\nSample query — avg risk score by zone:")
spark.sql("""
    SELECT zone,
           ROUND(AVG(risk_score), 1)       AS avg_risk,
           ROUND(AVG(congestion_score), 1) AS avg_congestion,
           COUNT(*)                        AS event_count
    FROM traffic_features
    GROUP BY zone
    ORDER BY avg_risk DESC
""").show()

print("\nSample query — severity distribution:")
spark.sql("""
    SELECT severity_label,
           COUNT(*) AS count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM traffic_features
    GROUP BY severity_label
    ORDER BY severity_label
""").show()

spark.stop()
