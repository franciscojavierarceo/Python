from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np

def generate_fake_data(num_records):
    # Generate fake data using pandas and numpy
    data = {
        'id': np.arange(1, num_records + 1),
        'name': ['Name_' + str(i) for i in range(1, num_records + 1)],
        'age': np.random.randint(18, 65, size=num_records),
        'salary': np.random.randint(30000, 100000, size=num_records)
    }
    df = pd.DataFrame(data)
    return df

def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName("SimpleSparkDemo").getOrCreate()

    # Generate fake data
    num_records = 100
    pandas_df = generate_fake_data(num_records)

    # Convert Pandas DataFrame to Spark DataFrame
    df = spark.createDataFrame(pandas_df)

    # Perform some transformations
    df_filtered = df.filter(col('age') > 30)
    df_avg_salary = df_filtered.groupBy().avg('salary')

    # Show the results
    print("Filtered Data:")
    df_filtered.show(5)
    print("Average Salary of People Older Than 30:")
    df_avg_salary.show()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()

