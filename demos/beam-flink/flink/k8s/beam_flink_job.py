import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.transforms.window as window


def run():
    options = PipelineOptions(
        [
            "--runner=FlinkRunner",
            "--flink_version=1.10",
            "--flink_master=localhost:8081",
            "--environment_type=EXTERNAL",
            "--environment_config=localhost:50000",
        ]
    )

    with beam.Pipeline(options=options) as p:
        (
            p
            | "Create words" >> beam.Create(["to be or not to be"])
            | "Split words" >> beam.FlatMap(lambda words: words.split(" "))
            | "Write to file" >> WriteToText("test.txt")
        )


if __name__ == "__main__":
    run()
