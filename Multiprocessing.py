from multiprocessing import Pool

def f(x):
    for _ in xrange(10**10):
        x += 1
    return x

if __name__ == '__main__':
    p = Pool(8)
    print(p.map(f, xrange(16)))
    

map(lambda x: x+"________cool", glob("*"))