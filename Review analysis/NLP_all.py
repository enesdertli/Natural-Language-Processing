import matplotlib.pyplot as plt
import math as math
from functools import reduce

accuracies = []


# Positive Reviews
import os;
with open("./tr_polarity.pos", 'rb') as f:
     # reviews_pos = f.read().decode('iso-8859-9').replace('\r', '');
     reviews_pos = f.read().decode('cp1254').replace('\r', '').splitlines();

# Negative Reviews
with open("./tr_polarity.neg", 'rb') as f:
     reviews_neg = f.read().decode('cp1254').replace('\r', '').splitlines();

# Check Dataset Size Equality
assert(len(reviews_pos) == len(reviews_neg))

# Slice the dataset for testing
data_size = len(reviews_pos);
test_size = round(data_size * 0.1); # 10% of whole dataset for each round.
print("Data size = %d, Test Size = %d, Last Index = %d" % (data_size, test_size, test_size * 10));
test_index = [(x * test_size, x * test_size + test_size) for x in range(10)];
print("Dataset slices for testing: %s" % test_index)

print(test_index[9][0])

# Utility function for dataset splitting. In the for loop, our first test set is the first 10% of the dataset, then it will be the next 10% and so on.
def split_dataset(data, split_indexes):
  data_test  = data[split_indexes[0]:split_indexes[1]];
  data_training = data[0:split_indexes[0]] + data[split_indexes[1]:len(data)];
  return { "test": data_test, "train": data_training};
        

import nltk;
nltk.download('punkt')

# Tokenize
reviews_pos_tokens = [nltk.word_tokenize(doc) for doc in reviews_pos];
reviews_neg_tokens = [nltk.word_tokenize(doc) for doc in reviews_neg];







# Validate test split construction
for i in range(10):
    splits = split_dataset(reviews_pos_tokens, test_index[i]);
    pos_test = splits['test'];
    pos_train = splits['train'];
    print("Pos test size: %d, \n Last item : %s" % (len(pos_test), pos_test[-1]) )
    print("Pos train size: %d, \n Last item : %s" % (len(pos_train), pos_train[0]) )

    splits = split_dataset(reviews_neg_tokens, test_index[0]);
    neg_test = splits['test'];
    neg_train = splits['train'];
    print("Neg test size: %d, \n Last item : %s" % (len(neg_test), neg_test[-1]) )
    print("Neg train size: %d, \n Last item : %s" % (len(neg_train), neg_train[0]) )
    # Vocabulary
    pos_train_flat = [X for x in pos_train for X in x];
    V_pos = { x for x in pos_train_flat }; 
    print("|V_pos| = %d, \n elements: %s" % (len(V_pos), V_pos))
    # 1-gram statistics, i.e. 1-gram LM for POS
    N = len(V_pos);
    LM_pos = {x:0 for x in V_pos};
    for token in pos_train_flat:
      LM_pos[token] += 1;

    # Most frequent tokens
    mfreq_pos = [(x,LM_pos[x]) for x in LM_pos];
    mfreq_pos.sort(key=lambda t: t[1], reverse=1);

    print("Most frequent 10 tokens: %s" % (mfreq_pos[:20]))

    # Vocabulary
    neg_train_flat = [X for x in neg_train for X in x];
    V_neg = { x for x in neg_train_flat }; 
    print("|V| = %d, \n elements: %s" % (len(V_neg), V_neg))

    # 1-gram statistics, i.e. 1-gram LM for Neg
    N = len(V_neg);
    LM_neg = {x:0 for x in V_neg};
    for token in neg_train_flat:
      LM_neg[token] += 1;

    # Most frequent tokens
    mfreq_neg = [(x,LM_neg[x]) for x in LM_neg];
    mfreq_neg.sort(key=lambda t: t[1], reverse=1);

    print("Most frequent 10 tokens: %s" % (mfreq_neg[:20]))

    #first 10 elements
    print('Vocabulary: ', list(V_pos)[:10])
    vocab = V_pos

    for idx, token in enumerate(vocab):
      print('index = %d \t vocabulary term: %s' % (idx, token))

      term2idx = {}
    for idx, token in enumerate(vocab):
        term2idx.update({token: idx})

    print(term2idx)

    #Indexes for the terms of individual documents. Documents are the test reviews.

    print('Term indexes for Review 1: %s' % [term2idx.get(token) for token in   pos_test[0]])

    # To access terms via index, create a new dictionary from index to terms
    idx2term = {}
    for term in term2idx:
        idx = term2idx.get(term)
        idx2term.update({idx: term})
    print(idx2term)

    # list of document vectors which is includes 0 or 1 for the train positive  reviews.
    # 1 means the term is in the document, 0 means the term is not in the document.
    pos_train_vectors = []
    for doc in pos_train:
        doc_vector = [0] * len(vocab)
        for token in doc:
            idx = term2idx.get(token)
            doc_vector[idx] = 1
        pos_train_vectors.append(doc_vector)
    print('Number of documents in the training set: %d' % len(pos_train_vectors))
    print('Number of terms in the vocabulary: %d' % len(vocab)) 
    print(len(pos_train_vectors))
    print(pos_train_vectors[0])

    print(pos_train_vectors[5][228], pos_train_vectors[3][227], pos_train_vectors[1]    [227])

    # create a new vector by averaging all vectors in the pos_train_vectors
    pos_train_vector_avg = [0] * len(vocab)
    for doc_vector in pos_train_vectors:
        for idx, val in enumerate(doc_vector):
            pos_train_vector_avg[idx] += val
    for idx, val in enumerate(pos_train_vector_avg):
        pos_train_vector_avg[idx] = val / len(pos_train_vectors)

    #Check wherher the test reviews contain anu unknown words!!!

    pos_test_flat = [w for doc in pos_test for w in doc]
    V_pos_test = set(pos_test_flat)
    print('Number of unique words in test set: %d' % len(V_pos_test))
    print(f'# of Unknow tokens = {len(V_pos_test.difference(V_pos))}, \n    {V_pos_test.difference(V_pos)} ')

    test_review = pos_test[1]
    print('Test review: %s' % test_review)

    V_test_review = set(test_review)
    print('Unknown tokens: %s' % V_test_review.difference(V_pos))

    # Calculate P(r|LM_pos)
    # Take p(w_i)|LM) as 1/n as a trivial approach to 'smoothing'
    score_pos = 0
    for token in test_review:
        score_pos += math.log10(LM_pos.get(token,1)/N)

    print('Score for positive review: %f' % score_pos)

    score_neg = 0
    for token in test_review:
        score_neg += math.log10(LM_neg.get(token,1)/N)
    print('Score for negative review: %f' % score_neg)

    print (score_pos > score_neg)

    def lm_scores(test_review):
      score_pos = 0;
      score_neg = 0;
      for token in test_review:
        score_pos += math.log10(LM_pos.get(token,1)/N);
        score_neg += math.log10(LM_neg.get(token,1)/N);
      return (score_pos, score_neg);




    pos_test_results = []; # 1 (TP): P(r|LM_pos) > P(r|LM_neg), otherwise 0 (FN)
    neg_test_results = []  # 1 (TN): P(r|LM_pos) < P(r|LM_neg), otherwise 0 (FP)

    for review in pos_test:
      scores = lm_scores(review);
      if scores[0] > scores[1]:
        pos_test_results.append(1);
      else:
        pos_test_results.append(0);

    TP = reduce(lambda x,y: x+y, pos_test_results);
    print("TP: %d,  FN: %d" % (TP, len(pos_test_results) - TP))

    for review in neg_test:
      scores = lm_scores(review);
      if scores[0] < scores[1]:
        neg_test_results.append(1);
      else:
        neg_test_results.append(0);

    TN = reduce(lambda x,y: x+y, neg_test_results);
    print("TN: %d,  FP: %d" % (TN, len(neg_test_results) - TN))
    print("Accuracy = %.2f" % ((TP+TN)/(len(pos_test_results)+len   (neg_test_results))))

    accuracy = (TP+TN)/(len(pos_test_results)+len(neg_test_results))

    accuracies.append(round(accuracy,3))
print(accuracies)






