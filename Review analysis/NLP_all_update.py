import matplotlib.pyplot as plt
import math as math
from functools import reduce


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
#print("Data size = %d, Test Size = %d, Last Index = %d" % (data_size, test_size, test_size * 10));
test_index = [(x * test_size, x * test_size + test_size) for x in range(10)];
#print("Dataset slices for testing: %s" % test_index)

#print(test_index[9][0])

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

resultList = []

# Validate test split construction
for i in range(10):
    splits = split_dataset(reviews_pos_tokens, test_index[i]);
    pos_test = splits['test'];
    pos_train = splits['train'];
    #print("Pos test size: %d, \n Last item : %s" % (len(pos_test), pos_test[-1]) )
    #print("Pos train size: %d, \n Last item : %s" % (len(pos_train), pos_train[0]) )

    splits = split_dataset(reviews_neg_tokens, test_index[i]);
    neg_test = splits['test'];
    neg_train = splits['train'];
    #print("Neg test size: %d, \n Last item : %s" % (len(neg_test), neg_test[-1]) )
    #print("Neg train size: %d, \n Last item : %s" % (len(neg_train), neg_train[0]) )

    # Vocabulary
    pos_train_flat = [X for x in pos_train for X in x];
    V_pos = { x for x in pos_train_flat }; 
    print("|V_pos| = %d, \n elements: %s" % (len(V_pos), V_pos))

    neg_train_flat = [X for x in neg_train for X in x];
    V_neg = { x for x in neg_train_flat };
    print("|V_neg| = %d, \n elements: %s" % (len(V_neg), V_neg))
    #! Positive Frequents
    # 1-gram statistics, i.e. 1-gram LM for POS
    N = len(V_pos);
    LM_pos = {x:0 for x in V_pos};
    for token in pos_train_flat:
      LM_pos[token] += 1;

    # Most frequent tokens
    mfreq_pos = [(x,LM_pos[x]) for x in LM_pos];
    mfreq_pos.sort(key=lambda t: t[1], reverse=1);

    #print("Most frequent 10 tokens: %s" % (mfreq_pos[:20]))
    #! Negative Frequents
    # 1-gram statistics, i.e. 1-gram LM for Neg
    N = len(V_neg);
    LM_neg = {x:0 for x in V_neg};
    for token in neg_train_flat:
      LM_neg[token] += 1;

    # Most frequent tokens
    mfreq_neg = [(x,LM_neg[x]) for x in LM_neg];
    mfreq_neg.sort(key=lambda t: t[1], reverse=1);

    #print("Most frequent 10 tokens: %s" % (mfreq_neg[:20]))


    vocab_pos = V_pos
    vocab_neg = V_neg
    #first 10 elements
    
    
    #! Positive term2idx
    for idx, token in enumerate(vocab_pos):
        print('index = %d \t vocabulary term: %s' % (idx, token))

    term2idx_pos = {}

    for idx, token in enumerate(vocab_pos):
        term2idx_pos.update({token: idx})
    #print(term2idx_pos)
    #! Negative term2idx
    for idx, token in enumerate(vocab_neg):
        print('index = %d \t vocabulary term: %s' % (idx, token))       

    term2idx_neg = {}

    for idx, token in enumerate(vocab_neg):
        term2idx_neg.update({token: idx})

    #print(term2idx_neg)

    #Indexes for the terms of individual documents. Documents are the test reviews.

    print('Term indexes for review 1 of positive test: %s' % [term2idx_pos.get(token) for token in pos_test[0]])
    print('Term indexes for review 1 of negative test: %s' % [term2idx_neg.get(token) for token in neg_test[0]])



    # To access terms via index, create a new dictionary from index to terms
    idx2term_pos = {}
    for term in term2idx_pos:
        idx = term2idx_pos.get(term)
        idx2term_pos.update({idx: term})
    #print(idx2term_pos)


    idx2term_neg = {}
    for term in term2idx_neg:
        idx = term2idx_neg.get(term)
        idx2term_neg.update({idx: term})
    #print(idx2term_neg)

    # list of document vectors which is includes 0 or 1 for the train positive  reviews.
    # 1 means the term is in the document, 0 means the term is not in the document.
    pos_train_vectors = []
    for doc in pos_train:
        doc_vector_pos = [0] * len(vocab_pos)
        for token in doc:
            idx = term2idx_pos.get(token)
            doc_vector_pos[idx] = 1
        pos_train_vectors.append(doc_vector_pos)
    print('Number of documents in the positive training set: %d' % len(pos_train_vectors))
    print('Number of terms in the positive vocabulary: %d' % len(vocab_pos)) 


    neg_train_vectors = []
    for doc in neg_train:
        doc_vector_neg = [0] * len(vocab_neg)
        for token in doc:
            idx = term2idx_neg.get(token)
            doc_vector_neg[idx] = 1
        neg_train_vectors.append(doc_vector_neg)
    print('Number of documents in the negative training set: %d' % len(neg_train_vectors))
    print('Number of terms in the negative vocabulary: %d' % len(vocab_neg))


    pos_test_vectors = []
    for doc in pos_test:
        doc_vector_pos = [0] * len(vocab_pos)
        for token in doc:
            if token in vocab_pos:
                idx = term2idx_pos.get(token)
                doc_vector_pos[idx] = 1
        pos_test_vectors.append(doc_vector_pos)

    neg_test_vectors = []
    for doc in neg_test:
        doc_vector_neg = [0] * len(vocab_neg)
        for token in doc:
            if token in vocab_neg:
                idx = term2idx_neg.get(token)
                doc_vector_neg[idx] = 1
        neg_test_vectors.append(doc_vector_neg)

    print(len(pos_test_vectors))
    print(len(neg_test_vectors))


    # create a new vector by averaging all vectors in the pos_train_vectors
    pos_train_vector_avg = [0] * len(vocab_pos)
    for doc_vector in pos_train_vectors:
        for idx, val in enumerate(doc_vector):
            pos_train_vector_avg[idx] += val
    for idx, val in enumerate(pos_train_vector_avg):
        pos_train_vector_avg[idx] = val / len(pos_train_vectors)

    # create a new vector bu averaging all vectors in the neg_train_vectors
    neg_train_vector_avg = [0] * len(vocab_neg)
    for doc_vector in neg_train_vectors:
        for idx, val in enumerate(doc_vector):
            neg_train_vector_avg[idx] += val
    for idx, val in enumerate(neg_train_vector_avg):
        neg_train_vector_avg[idx] = val / len(neg_train_vectors)


    result = []

    for doc_vector in pos_test_vectors:
        calculator_pos = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in pos_train_vector_avg]))
        calculator_neg = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in neg_train_vector_avg]))
        numerator_pos = sum([val1*val2 for val1, val2 in zip(doc_vector, pos_train_vector_avg)])
        numerator_neg = sum([val1*val2 for val1, val2 in zip(doc_vector, neg_train_vector_avg)])

        pos_ratio = numerator_pos / (calculator_pos + 1)
        neg_ratio = numerator_neg / (calculator_pos + 1)
        
        if pos_ratio > neg_ratio:
            result.append("1:1")
        else:
            result.append("0:1")

    for doc_vector in neg_test_vectors:
        calculator_pos = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in pos_train_vector_avg]))
        calculator_neg = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in neg_train_vector_avg]))
        numerator_pos = sum([val1*val2 for val1, val2 in zip(doc_vector, pos_train_vector_avg)])
        numerator_neg = sum([val1*val2 for val1, val2 in zip(doc_vector, neg_train_vector_avg)])


        pos_ratio = numerator_pos / (calculator_pos + 1)
        neg_ratio = numerator_neg / (calculator_neg + 1)


        if pos_ratio > neg_ratio:
            result.append("1:0")
        else:
            result.append("0:0")

    resultList.append(result)


for result in resultList:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(result)):
        if result[i] == "1:1":
            TP += 1
        elif result[i] == "0:0":
            TN += 1
        elif result[i] == "1:0":
            FP += 1
        elif result[i] == "0:1":
            FN += 1
    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)
    print("Accuracy: ", (TP+TN)/(TP+TN+FP+FN))
    print("Precision: ", TP/(TP+FP))
    print("Recall: ", TP/(TP+FN))
    print("F1: ", 2*TP/(2*TP+FP+FN))




