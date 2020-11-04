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


def rank_func(graph_view, i, damp=0.85):
    Lcin = graph_view.inputs_of(i, include_weights=False)
    sum = 0
    for j in Lcin:
        Jscore = graph_view.score_of(j)
        Jcout = graph_view.outputs_of(j, include_weights=False)
        sum += Jscore / len(Jcout)
    return (1 - damp) + damp * sum


if __name__ == "__main__":

    # try for various damping factors 
    damps = [ 0.50, 0.60, 0.70, 0.75, 0.85 ]
    for d in damps:
        pr = dx.PageRank(
            preprocs=[ word_tokenize, pos_tag, noun_and_adj_only ],
            link_func=dx.PrLinkMaker.proxy(),
            rank_func=dx.PrRanker.pagerank(damp=d)
        )
        pr.feed(sample)
        pr.set_inverse_uniform_scores()
        table = pr.execute()
        print(f"For damp factor {d}:")
        for i in range (4):
            print("[%.2f] %s" % (table[i][1], table[i][0]))