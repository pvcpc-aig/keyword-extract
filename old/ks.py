import io
import os
import sys
import math
import udax as dx
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path


useless_words = set(stopwords.words("english"))


# The INSPEC dataset is behind a paywall, so at the moment
# we will use the samples provided in the research paper.
samples = [
    [
        "BC-HurricaineGilbert, 09-11 339",
        "BC-Hurricaine Gilbert, 0348",
        "Hurricaine Gilbert heads toward Dominican Coast",
        "By Ruddy Gonzalez",
        "Associated Press Writer",
        "Santo Domingo, Dominican Republic (AP)",
        "Hurricaine Gilbert Swept towrd the Dominican Republic Sunday, and the Civil Defense alerted its heavily populated south coast to prepare for high winds, heavy rains, and high seas.",
        "The storm was approaching from the southeast with sustained winds of 75 mph gusting to 92 mph.",
        "\"There is no need for alarm,\" Civil Defense Director Eugenio Cabral said in a television alert shortly after midnight Saturday."
        "Cabral said residents of the province of Barahona should closely follow Gilbert’s movement.",
        "An estimated 100,000 people live in the province, including 70,000 in the city of Barahona, about 125 miles west of Santo Domingo.",
        "Tropical storm Gilbert formed in the eastern Carribean and strenghtened into a hurricaine Saturday night.",
        "The National Hurricaine Center in Miami reported its position at 2 a.m. Sunday at latitude 16.1 north, longitude 67.5 west, about 140 miles south of Ponce, Puerto Rico, and 200 miles southeast of Santo Domingo.",
        "The National Weather Service in San Juan, Puerto Rico, said Gilbert was moving westard at 15 mph with a \"broad area of cloudiness and heavy weather\" rotating around the center of the storm.",
        "The weather service issued a flash flood watch for Puerto Rico and the Virgin Islands until at least 6 p.m. Sunday.",
        "Strong winds associated with the Gilbert brought coastal flooding, strong southeast winds, and up to 12 feet to Puerto Rico’s south coast.",
        "There were no reports on casualties.",
        "San Juan, on the north coast, had heavy rains and gusts Saturday, but they subsided during the night.",
        "On Saturday, Hurricane Florence was downgraded to a tropical storm, and its remnants pushed inland from the U.S. Gulf Coast.",
        "Residents returned home, happy to find little damage from 90 mph winds and sheets of rain.",
        "Florence, the sixth named storm of the 1988 Atlantic storm season, was the second hurricane.",
        "The first, Debby, reached minimal hurricane strength briefly before hitting the Mexican coast last month."
    ]
]


def similarity(Iwords, Jwords): # i & j are both list of words representing a sentence
    common_words = []
    for word in Iwords:
        if word in Jwords:
            common_words.append(word)
    return len(common_words) / math.log10(len(Iwords) * len(Jwords))


def gen_graph(sample, defscore=1):
    """
    Graph is a list of tuple-like lists with the format:

        T[0] - sentence index in the sample
        T[1] - the score of the sentence
        T[2] - list of inward connections represented by sentence indices.
        T[3] - list of inward weights corresponding to some connections.
        T[4] - list of outward connections represented by sentence indices.
        T[5] - list of outward weights corresponding to some connections.
    """
    graph = []

    def _link_sentences(i, j, weight):
        nonlocal graph

        _, _, Icin, Iwin, Icout, Iwout = graph[i]
        _, _, Jcin, Jwin, Jcout, Jwout = graph[j]

        if j not in Icin:
            Icin.append(j)
            Iwin.append(weight)
        if j not in Icout:
            Icout.append(j)
            Iwout.append(weight)

        if i not in Jcin:
            Jcin.append(i)
            Jwin.append(weight)
        if i not in Jcout:
            Jcout.append(i)
            Jwout.append(weight)

    for i in range(len(sample)):
        graph.append([i, defscore, [], [], [], []])

    for i in range(len(sample)):
        i_sent = sample[i]
        for j in range(len(sample)):
            j_sent = sample[j]
            _link_sentences(i, j, similarity(i_sent, j_sent))
    
    return graph


def rankw(G, i, damp=0.85, visited=None):
    """
    An implementation of the weighted score function as described in the
    paper. The structure expected of the graph G is described by `gen_graph`
    above.
    """
    _, _, Lcin, Lwin, _, _ = G[i]

    sumr = 0
    for j, w in zip(Lcin, Lwin):
        _, Jscore, _, _, Jcout, Jwout = G[j]
        sumr += Jscore * w / sum(Jwout)
    return (1 - damp) + damp * sumr


def rank_sample(sample, convthresh=1e-4, sumlen=5):
    # preprocess the sample first
    print("Sanitizing sample...")
    sanitized_sample = []
    for sent in sample:
        if len(sent.strip()) == 0:
            continue
        tokenized = word_tokenize(sent)
        filtered = filter(lambda x: x not in useless_words, tokenized)
        extracted = list(filtered)
        sanitized_sample.append(extracted)
    
    # generate and process graph
    print("Generating graph...")
    graph = gen_graph(sanitized_sample, defscore=1/len(sanitized_sample))
    rescore_buffer = [ x[1] for x in graph ]

    print("Processing...")
    iterations = 0
    while True:
        iterations += 1
        min_err = 1
        for i, e in enumerate(graph):
            n_score = rankw(graph, i)
            err = n_score - rescore_buffer[i]
            rescore_buffer[i] = n_score
            if err < min_err:
                min_err = err
            if min_err <= convthresh: # the paper says "any node", I wonder if that's correct
                break
        for i, s in enumerate(rescore_buffer):
            graph[i][1] = s
        if min_err <= convthresh:
            break
    
    print(f"Reached conversion after {iterations} iterations.")

    # Sort the graph by the computed score in reverse order to get the
    # highest ranked sentences first.
    table = sorted(graph, key=lambda x: x[1], reverse=True)
    # for t in table:
    #     print(t[1], sample[t[0]])
    converted_table = [ sample[x[0]] for x in table ]

    sumres = io.StringIO()
    for i in range(sumlen):
        sumres.write(f"{converted_table[i]} ")

    return converted_table, sumres.getvalue()


if __name__ == "__main__":
    for i, sample in enumerate(samples):
        table, summary = rank_sample(sample)
        print(f"Summary for sample {i}:")
        print(summary)