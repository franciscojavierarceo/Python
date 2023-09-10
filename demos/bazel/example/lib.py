import sys


def hello_world() -> None:
    print("hello world")
    print("we are using python version {}".format(sys.version))


def get_three() -> int:
    return 3


def main() -> None:
    hello_world()


if __name__ == "__main__":
    main()
