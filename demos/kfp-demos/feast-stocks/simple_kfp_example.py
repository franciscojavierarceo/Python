from kfp import local
from kfp import dsl
from kfp.dsl import Output, Artifact
import json

local.init(runner=local.SubprocessRunner())

@dsl.component
def add(a: int, b: int, out_artifact: Output[Artifact]):
    import json

    result = json.dumps(a + b)

    with open(out_artifact.path, 'w') as f:
        f.write(result)

    out_artifact.metadata['operation'] = 'addition'


task = add(a=1, b=2)
task2 = add(a=1, b=3)
# can read artifact contents
with open(task.outputs['out_artifact'].path) as f:
    contents = f.read()

with open(task2.outputs['out_artifact'].path) as f2:
    contents2 = f2.read()

assert json.loads(contents) == 3
assert task.outputs['out_artifact'].metadata['operation'] == 'addition'

assert json.loads(contents2) == 4
assert task2.outputs['out_artifact'].metadata['operation'] == 'addition'
