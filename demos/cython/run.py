import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
from cython_demo import calcScore

def main():
    group = 'A'
    tags = ['t1', 't2', 't3']
    d = {
        'A': {
            't1': 1,
            't2': 2,
            't3': 3,
        }
    }
    result = calcScore(group, tags, d)
    print(result)


if __name__ == '__main__':
    main()
