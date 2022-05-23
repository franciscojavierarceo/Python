# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:08:25 2015

@author: franciscojavierarceo
"""
import re
import sys
import os
import gzip
import networkx as nx

wd = "/Users/franciscojavierarceo/Downloads"
os.chdir(wd)

fh = gzip.open("knuth_miles.txt.gz", "r")


def miles_graph(fh):
    G = nx.Graph()
    G.position = {}
    G.population = {}
    cities = []
    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):  # skip comments
            continue

        numfind = re.compile("^\d+")

        if numfind.match(line):  # this line is distances
            dist = line.split()
            for d in dist:
                G.add_edge(city, cities[i], weight=int(d))
                i = i + 1

        else:  # this line is a city, position, population
            i = 1
            (city, coordpop) = line.split("[")
            cities.insert(0, city)
            (coord, pop) = coordpop.split("]")
            (y, x) = coord.split(",")

            G.add_node(city)
            # assign position - flip x axis for matplotlib, shift origin
            G.position[city] = (-int(x) + 7500, int(y) - 3000)
            G.population[city] = float(pop) / 1000.0
    return G


G = miles_graph(fh)

H = nx.Graph()
for v in G:
    H.add_node(v)
for (u, v, d) in G.edges(data=True):
    if d["weight"] < 300:
        H.add_edge(u, v)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
# with nodes colored by degree sized by population
node_color = [float(H.degree(v)) for v in H]
nx.draw(
    H,
    G.position,
    node_size=[G.population[v] for v in H],
    node_color=node_color,
    with_labels=False,
)
# scale the axes equally
plt.xlim(-5000, 500)
plt.ylim(-2000, 3500)
