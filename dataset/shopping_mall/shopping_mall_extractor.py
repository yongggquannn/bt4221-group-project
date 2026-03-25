import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["SPARK_SUBMIT_OPTS"] = (
    "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens java.base/java.lang=ALL-UNNAMED "
    "--add-opens java.base/java.util=ALL-UNNAMED"
)

spark = (
    SparkSession.builder
    .appName("shopping_mall_cleaner")
    .config("spark.driver.extraJavaOptions",
            "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
            "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens java.base/java.lang=ALL-UNNAMED "
            "--add-opens java.base/java.util=ALL-UNNAMED "
            "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions",
            "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
            "-Djava.security.manager=allow")
    .getOrCreate()
)

def clean_and_save_malls():
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "singapore_malls_pois.csv")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shopping_malls.csv")

    df = spark.read.option("header", True).csv(input_path)
    # Clean: trim whitespace, drop rows with empty or null name, deduplicate by name
    df = df.withColumn("name", F.trim(F.col("name")))
    df = df.filter(F.col("name").isNotNull() & (F.col("name") != ""))
    df = df.dropDuplicates(["name"])
    # Optionally, drop rows with missing lat/lon if required:
    # df = df.filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())

    logger.info("Writing %d cleaned mall records to %s", df.count(), output_path)
    df.toPandas().to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_and_save_malls()
