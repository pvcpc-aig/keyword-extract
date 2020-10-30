"""
An implementation of the TextRank algorithm used to extract words
as specified in |TextRank: Bringing Order into Texts|, by Rada
Mihalcea, Paul Tarau 
(https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

This implementation focuses specifically on keyword ranking from
some text, `ks.py` focuses on sentence ranking.
"""
import os
import sys
import random
import udax as dx
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from pathlib import Path


# The INSPEC dataset is behind a paywall, so at the moment
# we will use the samples provided in the research paper.
samples = [
    """
    Compatibility of systems of linear constraints over the set of natural numbers.
    Criteria of compatibility of a system of linear Diophantine equations, strict
    inequations, and nonstrict inequations are considered. Upper bounds for
    components of a minimal set of solutions and algorithms of construction of
    minimal generating sets of solutions for all types of systems are given.
    These criteria and the corresponding algorithms for constructing a minimal
    supporting set of solutions can be used in solving all considered types
    systems and systems of mixed types.
    """
]


def is_noun_or_adj(word_info):
    """
    A filtered used in `rank_sample` to extract words that are
    only nouns or adjectives as tagged by NLTK. This follows the
    suggestion of the paper to use only nouns and adjectives.
    """
    word, pos = word_info
    return pos.startswith("NN") or pos.startswith("JJ")


def gather_proxy(L, i, radius=2):
    """
    Collects all of the elements around the index `i` in list `L` that
    are within `radius` indices of itself. This does not include the
    index `i` itself.
    
    The elements are returned in their listed order.
    """
    low = i - radius
    high = i + radius
    result = []

    j = low
    while j <= high:
        if j >= len(L):
            break
        if 0 <= j and j != i:
            result.append(L[j])
        j += 1
    return result


def gen_graph(words, defscore=1, cofact=2):
    """
    Generates a graph of words as described in the paper for keyword
    extraction with a given `cofact`, co-occurance factor, which is
    the maximum proximity of words that may be linked together.

    NOTE: The filtering should be done prior to invoking this function.

    A tuple-like list is returned containing a dict and a list in that order. 
    The list is the graph itself, containing tuple-list lists with format:

        T[0] - A unique word in the graph.
        T[1] - The current score of the word.
        T[2] - The list of inward directed indices.
        T[3] - The list of outward directed indices.
    
    The dict is a mapping of all of the unique words to their index
    in the list pseudo-graph.
    """
    map = {}
    graph = []

    def _push_word(word):
        nonlocal map, graph

        if word in map:
            return map[word]
        
        index = len(graph)
        map[word] = index
        graph.append([word, defscore, [], []])
        return index
    
    def _link_words(i, j): # i - index of the parent word, j - index of the child word
        nonlocal map, graph

        Lout = graph[i][3]
        Lin = graph[j][2]

        if j not in Lout:
            Lout.append(j)

        if i not in Lin:
            Lin.append(i)

    for i, word in enumerate(words):
        main_i = _push_word(word)
        proxies = gather_proxy(words, i, radius=cofact)
        for proxy in proxies:
            proxy_i = _push_word(proxy)
            _link_words(main_i, proxy_i)
    
    return map, graph


def rank(G, i, damp=0.85, visited=None):
    """
    An implementation of the unweighted score function as described
    in the paper given the graph `G` and the node index `i`. The
    graph is expected to be a list of tuple-like lists with structure:

        T[0] - A unique word in the graph.
        T[1] - The current score of the word.
        T[2] - The list of inward directed indices.
        T[3] - The list of outward directed indices.
    """
    # if visited is None:
    #     visited = set()

    word, score, Lin, Lout = G[i]
    # if i in visited:
    #     return score
    # visited.add(i)

    sum = 0
    for j in Lin:
        _, Jscore, _, Jout = G[j]
        # sum += rank(G, j, visited=visited) / len(Jout)
        sum += Jscore / len(Jout)
    return (1 - damp) + damp * sum


def rank_sample(sample, convthresh=1e-4):
    # preprocess the sample text first
    print("Filtering sample...")
    # tokenized = word_tokenize(sample)
    tokenized = dx.s_norm(sample).split()
    tagged = pos_tag(tokenized)
    filtered = filter(is_noun_or_adj, tagged)
    extracted = [ x[0] for x in filtered ]

    # generate graph and process it
    print("Generating graph...")
    map, graph = gen_graph(extracted)
    if len(graph) > 0:
        rank(graph, 0)

    def _score(stop_on_conv=True):
        nonlocal convthresh, graph

        min_err = 1
        rescore_buffer = [ x[1] for x in graph ]
        for i, e in enumerate(graph):
            n_score = rank(graph, i)
            err = n_score - rescore_buffer[i]
            rescore_buffer[i] = n_score
            if err < min_err:
                min_err = err
            if stop_on_conv and err <= convthresh:
                break
        
        for i, s in enumerate(rescore_buffer):
            graph[i][1] = s

        return min_err

    print("Processing...")
    iterations = 1
    _score(stop_on_conv=False)
    while True:
        iterations += 1
        if _score() <= convthresh:
            break
    
    print(f"Reached conversion after {iterations} iterations.")

    # We sort the graph by the newly computed score, and reverse
    # it so that the highest score is first.
    ranked = sorted(graph, key=lambda x: x[1], reverse=True)
    # for r in ranked:
    #     print(r)
    return [ x[0] for x in ranked ]


if __name__ == "__main__":
    for i, sample in enumerate(samples):
        keywords = rank_sample(sample)
        print(f"Top 5 keywords in sample {i+1}:")
        for j in range(5):
            print(f"{j+1}. {keywords[j]}")