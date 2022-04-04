import apache_beam as beam

inputs = [
    ('🐹', '🌽'),
    ('🐼', '🎋'),
    ('🐰', '🥕'),
    ('🐹', '🌰'),
    ('🐰', '🥒'),
]

with beam.Pipeline() as pipeline:
    outputs = (
            pipeline
            | 'Create (animal, food) pairs' >> beam.Create(inputs)
            | 'Group foods by animals' >> beam.GroupByKey()
    )

    outputs | beam.Map(print)