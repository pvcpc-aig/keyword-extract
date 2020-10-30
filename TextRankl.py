from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
import stanza
import string

word_id_reverse = {}
word_id = {}
word_pos = {}

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

lemmatizer = WordNetLemmatizer()

def extract_nn_jj (text):
    global word_id_reverse , word_id, word_pos
    word_id_reverse = {}
    word_id = {}
    word_pos = {}
    stanza_tags = nlp(text.lower())
    text = word_tokenize(text.lower())
    tags = []
    for sentence in stanza_tags.sentences:
        for word in sentence.words:
            tags.append((word.text, word.xpos))
    tags = [(lemmatizer.lemmatize(tag[0]), tag[1]) for tag in tags]
    
    i = 0
    j = 0
    
    for word, tag in tags:
        if tag[:2] == 'NN' or tag[:2] == 'JJ':
            if not word in word_id:
                word_id[word] = j
                word_id_reverse[j] = word
                j += 1
                
            if word not in word_pos:
                word_pos[word] = [i]
            else:
                word_pos[word].append(i)
        i += 1
    
    
    adj = [[] for i in range(j)]
    N = 2
    for root, root_pos in word_pos.items():
        for neighbor, neighbor_pos in word_pos.items():
            if neighbor != root:
                c_N = float('inf')
                
                for k in root_pos:
                    for l in neighbor_pos:
                        c_N = min(c_N, abs(k - l))
                
                if c_N <= N:
                    adj[word_id[root]].append(word_id[neighbor])
    return adj
            
connectivity_graph = extract_nn_jj(open('data/abstracts/abstract1.txt').read())            

def text_rank_algorithm (adj):
    d = 0.85
    score = [1 for i in range(len(adj))]
    for i in range(100):
        score_change = 0
        for j in range(len(adj)):
            current_score = 0
            current_node = j
            neighbors = adj[current_node]
            for neighbor in neighbors:
                current_score += (1/len(adj[neighbor])) * score[neighbor]
            current_score *= d
            current_score = (1 - d) + current_score
            score_change += abs(score[current_node] - current_score)
            score[current_node] = current_score
        
        if score_change < 0.00001:
            break
    return score
    

scores = text_rank_algorithm(connectivity_graph)
scores = sorted([(scores[i], i) for i in range(len(scores))], reverse=True)

for score in scores:
    print(score[0], word_id_reverse[score[1]])