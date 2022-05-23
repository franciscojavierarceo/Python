import time
from datetime import datetime
from pprint import pprint

from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="example_python_operator",
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    # [START howto_operator_python]
    def print_context(ds, **kwargs):
        """Print the Airflow context and ds variable from the context."""
        pprint(kwargs)
        print(ds)
        return "Whatever you return gets printed in the logs"

    run_this = PythonOperator(
        task_id="print_the_context",
        python_callable=print_context,
    )
    # [END howto_operator_python]

    # [START howto_operator_python_kwargs]
    def my_sleeping_function(random_base):
        """This is a function that will run within the DAG execution"""
        time.sleep(random_base)

    # Generate 5 sleeping tasks, sleeping from 0.0 to 0.4 seconds respectively
    for i in range(5):
        task = PythonOperator(
            task_id="sleep_for_" + str(i),
            python_callable=my_sleeping_function,
            op_kwargs={"random_base": float(i) / 10},
        )

        run_this >> task
    # [END howto_operator_python_kwargs]
