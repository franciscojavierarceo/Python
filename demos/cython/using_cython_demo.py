import time
from cython_demo import getScoresbyTag


def main():
    gs = {
        "Focus and Attention": [
            ("1 - being understood", 0.5),
            ("Confusion", 1),
            ("Comparison", 0),
            ("Following directions", 0.5),
            ("Focus", 1),
            ("ADHD", 0),
            ("Executive Functioning Issues", 1),
        ],
        "Strategies and Tips": [
            ("Strategies and tips", 1),
            ("Step by step / action list", 1),
            ("Downloadables and worksheets", 0.5),
            ("Infographic", 0.5),
            ("Video", 0),
        ],
    }
    gd = {"Confusion": 1.25, "Focus": 1.25, "Step by step / action list": 0.75}
    articles = {
        "article_1": ["Confusion"],
        "article_2": ["Focus", "Confusion"],
        "article_3": ["Focus", "Confusion", "Step by step / action list"],
        "article_4": ["Step by step / action list"],
    }
    tags = ["Focus and Attention", "Strategies and Tips"]
    fin = getScoresbyTag(articles, gs, gd, tags)
    for i in fin:
        print(i)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
