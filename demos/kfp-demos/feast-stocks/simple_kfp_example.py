from kfp import local
from kfp import dsl
from kfp.dsl import Output, Artifact
import json

local.init(runner=local.SubprocessRunner())


@dsl.component
def add(a: int, b: int, out_artifact: Output[Artifact]) -> None:
    import json

    result = json.dumps(a + b)

    with open(out_artifact.path, "w") as f:
        f.write(result)

    out_artifact.metadata["operation"] = "addition"


@dsl.component
def verify_result(artifact: Artifact, expected_value: int) -> None:
    import json

    with open(artifact.path) as f:
        contents = f.read()

    result = json.loads(contents)
    assert result == expected_value
    assert artifact.metadata["operation"] == "addition"


@dsl.pipeline(
    name="Simple pipeline",
    description="Some basic tests",
)
def add_pipeline() -> None:
    task1 = add(a=1, b=2)
    task2 = add(a=1, b=3)

    verify_task1 = verify_result(
        artifact=task1.outputs["out_artifact"], expected_value=3
    )
    verify_task2 = verify_result(
        artifact=task2.outputs["out_artifact"], expected_value=4
    )

    verify_task1.after(task1)
    verify_task2.after(task2)


# Compile and run the pipeline
if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(add_pipeline, "add_pipeline.yaml")

    # Execute the pipeline directly
    add_pipeline()
