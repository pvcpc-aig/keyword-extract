import udax as dx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


sample = \
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


def noun_and_adj_only(tagged):
    result = []
    for data in tagged:
        word, pos = data
        if pos.startswith("NN") or pos.startswith("JJ"):
            result.append(word)
    return result


def link_func(linker):
    N = 2
    j_tokens = linker.inbound_tokens
    for i, _ in enumerate(j_tokens):
        low = max(0, i - N)
        high = min(len(j_tokens) - 1, i + N)
        while low <= high:
            if low == i:
                low += 1
                continue
            yield linker.inbound(i, low)
            low += 1


def rank_func(graph_view, i, damp=0.85):
    Lcin = graph_view.inputs_of(i, include_weights=False)
    sum = 0
    for j in Lcin:
        Jscore = graph_view.score_of(j)
        Jcout = graph_view.outputs_of(j, include_weights=False)
        sum += Jscore / len(Jcout)
    return (1 - damp) + damp * sum


if __name__ == "__main__":
    pr = dx.PageRank(
        preprocs=[ word_tokenize, pos_tag, noun_and_adj_only ],
        link_func=link_func,
        rank_func=rank_func)
    pr.feed(sample)
    pr.set_inverse_uniform_scores()
    table = pr.execute()
    
    for t in table:
        print(t[0], t[1])