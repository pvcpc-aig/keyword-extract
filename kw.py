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
from nltk.corpus import stopwords
from pathlib import Path


useless_words = set(stopwords.words("english"))


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


def gather_proxy(L, i, radius=3):
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

        _, _, Iin, Iout = graph[i]
        _, _, Jin, Jout = graph[j]

        # this will effectively create an undirected graph, simulated
        # by bidirection.
        if j not in Iin:
            Iin.append(j)
        if j not in Iout:
            Iout.append(j)

        if i not in Jin:
            Jin.append(i)
        if i not in Jout:
            Jout.append(i)

    for i, word in enumerate(words):
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
    _, _, Lin, _ = G[i]

    sum = 0
    for j in Lin:
        _, Jscore, _, Jout = G[j]
        sum += Jscore / len(Jout)
    return (1 - damp) + damp * sum


def rank_sample(sample, convthresh=1e-4):
    # preprocess the sample text first
    print("Filtering sample...")
    # tokenized = dx.s_norm(sample).split()
    tokenized = word_tokenize(sample)
    tagged = pos_tag(tokenized)
    filtered = filter(is_noun_or_adj, tagged)
    extracted = [ x[0] for x in filtered ]
    # extracted = list(filter(lambda x: x not in useless_words, tokenized))

    # generate graph and process it
    print("Generating graph...")
    map, graph = gen_graph(extracted, defscore=1/len(extracted))
    rescore_buffer = [ x[1] for x in graph ]

    print("Processing...")
    iterations = 0
    while True:
        iterations += 1
        min_err = 1
        for i, e in enumerate(graph):
            n_score = rank(graph, i)
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

    # We sort the graph by the newly computed score, and reverse
    # it so that the highest score is first.
    table = sorted(graph, key=lambda x: x[1], reverse=True)
    table_set = set([ x[0] for x in table ])

    # Collapse multi-word keywords into a single entry:
    collapsed = []
    i = 0
    while i < len(tokenized):
        word = tokenized[i]
        if word in table_set:
            streak = [ word ]
            j = i + 1
            while j < len(tokenized):
                n_word = tokenized[j]
                if n_word not in table_set:
                    break
                streak.append(n_word)
                j += 1

            if len(streak) > 1:
                i += j - 1 - i
                collapsed.append(' '.join(streak))
        i += 1
    
    return table, collapsed


if __name__ == "__main__":
    for i, sample in enumerate(samples):
        table, collapsed = rank_sample(sample)

        print(f"Top keywords in sample {i+1}:")
        for j in range(25):
            print(f"{j+1}. {table[j][0]}")

        print(f"Collapsed keywords in sample {i+1}:")
        for j, e in enumerate(collapsed):
            print(f"{j+1}. {e}")