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



import nltk;
nltk.download('punkt')


resultList = []
resultListImprove = []

#! Tokenize

def tokenization(reviews_pos, reviews_neg):
    reviews_pos_tokens = [nltk.word_tokenize(doc) for doc in reviews_pos];
    reviews_neg_tokens = [nltk.word_tokenize(doc) for doc in reviews_neg];
    return reviews_pos_tokens, reviews_neg_tokens

# Utility function for dataset splitting. In the for loop, our first test set is the first 10% of the dataset, then it will be the next 10% and so on.
def split_dataset(data, split_indexes):
  data_test  = data[split_indexes[0]:split_indexes[1]];
  data_training = data[0:split_indexes[0]] + data[split_indexes[1]:len(data)];
  return { "test": data_test, "train": data_training};


def flat_pos():
    pos_train_flat = [X for x in pos_train for X in x]
    V_pos = { x for x in pos_train_flat }; 
    print("|V_pos| = %d, \n elements: %s" % (len(V_pos), V_pos))
    return V_pos, pos_train_flat


def flat_neg():
    neg_train_flat = [X for x in neg_train for X in x]
    V_neg = { x for x in neg_train_flat }; 
    print("|V_neg| = %d, \n elements: %s" % (len(V_neg), V_neg))
    return V_neg, neg_train_flat

def frequency_pos(V_pos, pos_train_flat):
    #N = len(V_pos);
    LM_pos = {x:0 for x in V_pos};
    for token in pos_train_flat:
      LM_pos[token] += 1;

    # Most frequent tokens
    mfreq_pos = [(x,LM_pos[x]) for x in LM_pos];
    mfreq_pos.sort(key=lambda t: t[1], reverse=1);
    return mfreq_pos

def frequency_neg(V_neg, neg_train_flat):
    #N = len(V_neg);
    LM_neg = {x:0 for x in V_neg};
    for token in neg_train_flat:
      LM_neg[token] += 1;

    # Most frequent tokens
    mfreq_neg = [(x,LM_neg[x]) for x in LM_neg];
    mfreq_neg.sort(key=lambda t: t[1], reverse=1);
    return mfreq_neg

def pos_train_vector_d():
    pos_train_vectors = []
    for doc in pos_train:
        doc_vector_pos = [0] * len(vocab_pos)
        for token in doc:
            idx = term2idx_pos.get(token)
            doc_vector_pos[idx] = 1
        pos_train_vectors.append(doc_vector_pos)
    print('Number of documents in the positive training set: %d' % len(pos_train_vectors))
    print('Number of terms in the positive vocabulary: %d' % len(vocab_pos))
    return pos_train_vectors

def neg_train_vector():
    neg_train_vectors = []
    for doc in neg_train:
        doc_vector_neg = [0] * len(vocab_neg)
        for token in doc:
            idx = term2idx_neg.get(token)
            doc_vector_neg[idx] = 1
        neg_train_vectors.append(doc_vector_neg)
    print('Number of documents in the negative training set: %d' % len(neg_train_vectors))
    print('Number of terms in the negative vocabulary: %d' % len(vocab_neg)) 
    return neg_train_vectors

def pos_test_vectors_d():
    pos_test_vectors = []
    for doc in pos_test:
        doc_vector_pos = [0] * len(vocab_pos)
        for token in doc:
            if token in vocab_pos:
                idx = term2idx_pos.get(token)
                doc_vector_pos[idx] = 1
        pos_test_vectors.append(doc_vector_pos)
    return pos_test_vectors

def neg_test_vectors_d():
    neg_test_vectors = []
    for doc in neg_test:
        doc_vector_neg = [0] * len(vocab_neg)
        for token in doc:
            if token in vocab_neg:
                idx = term2idx_neg.get(token)
                doc_vector_neg[idx] = 1
        neg_test_vectors.append(doc_vector_neg)
    return neg_test_vectors

def pos_train_vector_avg_d():
    pos_train_vector_avg = [0]* all_vocab
    for doc_vector in pos_train_vectors:
        for idx, val in enumerate(doc_vector):
            pos_train_vector_avg[idx] += val
    for idx, val in enumerate(pos_train_vector_avg):
        pos_train_vector_avg[idx] = val / len(pos_train_vectors)
    return pos_train_vector_avg

def neg_train_vector_avg_d():
    neg_train_vector_avg = [0] * all_vocab
    for doc_vector in neg_train_vectors:
        for idx, val in enumerate(doc_vector):
            neg_train_vector_avg[idx] += val
    for idx, val in enumerate(neg_train_vector_avg):
        neg_train_vector_avg[idx] = val / len(neg_train_vectors)
    return neg_train_vector_avg


def term2idx(vocab_pos, vocab_neg):
    term2idx_pos = {}
    for idx,token in enumerate(vocab_pos):
        term2idx_pos.update({token:idx})
    term2idx_neg = {}
    for idx,token in enumerate(vocab_neg):
        term2idx_neg.update({token:idx})
    return term2idx_pos, term2idx_neg

def idx2term():
    idx2term_pos = {}
    idx2term_neg = {}
    for term in term2idx_pos:
        idx = term2idx_pos.get(term)
        idx2term_pos.update({idx: term})
    for term in term2idx_neg:
        idx = term2idx_neg.get(term)
        idx2term_neg.update({idx: term})
    return idx2term_pos, idx2term_neg

def calculate_accuracy(pos_test_vectors, neg_test_vectors, pos_train_vector_avg, neg_train_vector_avg,result):
    
    for doc_vector in pos_test_vectors:
        denominator_pos = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in pos_train_vector_avg]))
        denominator_neg = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in neg_train_vector_avg]))
        numerator_pos = sum([val1*val2 for val1, val2 in zip(doc_vector, pos_train_vector_avg)])
        numerator_neg = sum([val1*val2 for val1, val2 in zip(doc_vector, neg_train_vector_avg)])

        pos_ratio = numerator_pos / (denominator_pos + 1)
        neg_ratio = numerator_neg / (denominator_pos + 1)
        
        if pos_ratio > neg_ratio:
            result.append("1:1")
        else:
            result.append("0:1")

    for doc_vector in neg_test_vectors:
        denominator_pos = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in pos_train_vector_avg]))
        denominator_neg = math.sqrt(sum([val**2 for val in doc_vector]))*math.sqrt(sum([val**2 for val in neg_train_vector_avg]))
        numerator_pos = sum([val1*val2 for val1, val2 in zip(doc_vector, pos_train_vector_avg)])
        numerator_neg = sum([val1*val2 for val1, val2 in zip(doc_vector, neg_train_vector_avg)])


        pos_ratio = numerator_pos / (denominator_pos + 1)
        neg_ratio = numerator_neg / (denominator_neg + 1)


        if pos_ratio > neg_ratio:
            result.append("1:0")
        else:
            result.append("0:0")

    resultList.append(result)

def improveTokenization(pos_test, neg_test, pos_train, neg_train):
    splits = split_dataset(reviews_pos_tokens, test_index[i]);
    pos_test = splits['test'];
    pos_train = splits['train'];

    splits = split_dataset(reviews_neg_tokens, test_index[i]);
    neg_test = splits['test'];
    neg_train = splits['train'];


    for doc in pos_test:
        for idx, token in enumerate(doc):
            doc[idx] = token.lower()
            if not token.isalnum():
                doc[idx] = token[:-1]
    for doc in neg_test:
        for idx, token in enumerate(doc):
            doc[idx] = token.lower()
            if not token.isalnum():
                doc[idx] = token[:-1]
    for doc in pos_train:
        for idx, token in enumerate(doc):
            doc[idx] = token.lower()
            if not token.isalnum():
                doc[idx] = token[:-1]
    for doc in neg_train:
        for idx, token in enumerate(doc):
            doc[idx] = token.lower()
            if not token.isalnum():
                doc[idx] = token[:-1]

    return pos_test, neg_test, pos_train, neg_train


# Validate test split construction
# We created a for loop to iterate through the test_index list and split the dataset into test and train sets. We will use one-hot model to represent the documents. Also we will create a vocabulary for each class. We will use the vocabulary to create a vector for each document. We will use the vector to calculate the cosine similarity between the test document and the average vector of the train documents. If the pozitive cosine similarity is greater than the negative cosine similarity, we will predict the document to be positive. If the negative cosine similarity is greater than the positive cosine similarity, we will predict the document to be negative. We will calculate the accuracy of the model and store it in a list. We will repeat the process for each test set and calculate the average accuracy of the model. And if our prediction is correct on True Positive, we will add 1:1 to the result list. If our prediction is correct on True Negative, we will add 0:0 to the result list. If our prediction is wrong on False Positive, we will add 1:0 to the result list. If our prediction is wrong on False Negative, we will add 0:1 to the result list.

for i in range(10):

    reviews_pos_tokens, reviews_neg_tokens = tokenization(reviews_pos, reviews_neg)
   
    splits = split_dataset(reviews_pos_tokens, test_index[i]);
    pos_test = splits['test'];
    pos_train = splits['train'];
    

    splits = split_dataset(reviews_neg_tokens, test_index[i]);
    neg_test = splits['test'];
    neg_train = splits['train'];
    

    # Vocabulary
    V_pos, pos_train_flat = flat_pos() 
    V_neg, neg_train_flat = flat_neg()
    

    #* Frequency
    #! Positive Frequents
    mfreq_pos = frequency_pos(V_pos, pos_train_flat)
    print("Most frequent 10 tokens: %s" % (mfreq_pos[:20]))

    #! Negative Frequents
    mfreq_neg =frequency_neg(V_neg, neg_train_flat)
    print("Most frequent 10 tokens: %s" % (mfreq_neg[:20]))
    

    vocab_pos = V_pos
    vocab_neg = V_neg
    all_vocab = len(vocab_neg) + len(vocab_pos)
    
    term2idx_pos, term2idx_neg = term2idx(vocab_pos, vocab_neg)

    #Indexes for the terms of individual documents. Documents are the test reviews.
    print('Term indexes for review 1 of positive test: %s' % [term2idx_pos.get(token) for token in pos_test[0]])
    print('Term indexes for review 1 of negative test: %s' % [term2idx_neg.get(token) for token in neg_test[0]])

    idx2term_pos, idx2term_neg = idx2term()


    pos_train_vectors = pos_train_vector_d()
    neg_train_vectors = neg_train_vector()
    pos_test_vectors = pos_test_vectors_d()
    neg_test_vectors = neg_test_vectors_d()


    pos_train_vector_avg = pos_train_vector_avg_d()
    neg_train_vector_avg = neg_train_vector_avg_d()


    result = []
    calculate_accuracy(pos_test_vectors, neg_test_vectors, pos_train_vector_avg, neg_train_vector_avg, result)
    resultList.append(result)
    
    
#We created a list name is all_accuracies_before_improving. Our resultList contains the result of each fold. We will use this list to calculate the accuracy of each fold. 
all_accuracies_before_improving = []

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
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    all_accuracies_before_improving.append(accuracy)


# To calculate the average accuracy of all folds, we will use the all_accuracies_before_improving list. If we calculate the TP+TN/TP+TN+FP+FN, we will get the accuracy of each fold. We will add all the accuracies and divide by the number of folds.
print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} ')
print("Accuracy: ", (TP+TN)/(TP+TN+FP+FN))
print("Precision: ", TP/(TP+FP))
print('Recall:', TP/(TP+FN))
print("F1: ", 2*TP/(2*TP+FP+FN))
print('\n')
avg_accuracy = sum(all_accuracies_before_improving)/len(all_accuracies_before_improving)
print('Average accuracy is: ',avg_accuracy)



#We can improve the accuracy by using the following methods:
#1. Stop words elimination
#2. Stemming
#3. Lemmatization
#4. TF-IDF
#5. Word2Vec
#6. Better tokenization

# We will use isalnum to remove punctuation and numbers because they are not useful for our model.


#! Improving the model by using isalnum
for i in range(10):
    
    splits = split_dataset(reviews_pos_tokens, test_index[i]);
    pos_test = splits['test'];
    pos_train = splits['train'];

    splits = split_dataset(reviews_neg_tokens, test_index[i]);
    neg_test = splits['test'];
    neg_train = splits['train'];

    
    pos_test_tokens, neg_test_tokens, pos_train_tokens, neg_train_tokens = improveTokenization(pos_test, neg_test, pos_train, neg_train)


    V_pos, pos_train_flat = flat_pos()
    V_neg, neg_train_flat = flat_neg()

    vocab_pos = V_pos
    vocab_neg = V_neg

    term2idx_pos, term2idx_neg = term2idx(vocab_pos, vocab_neg)
    idx2term_pos, idx2term_neg = idx2term()
    pos_train_vectors = pos_train_vector_d()
    neg_train_vectors = neg_train_vector()
    pos_test_vectors = pos_test_vectors_d()
    neg_test_vectors = neg_test_vectors_d()
    pos_train_vector_avg = pos_train_vector_avg_d()
    neg_train_vector_avg = neg_train_vector_avg_d()
    result = []
    calculate_accuracy(pos_test_vectors, neg_test_vectors, pos_train_vector_avg, neg_train_vector_avg, result)
    resultListImprove.append(result)
    
    break


all_accuracies_after_improving = []

for result in resultListImprove:
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
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    all_accuracies_after_improving.append(accuracy)

print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} ')
print("Accuracy: ", (TP+TN)/(TP+TN+FP+FN))
print("Precision: ", TP/(TP+FP))
print('Recall:', TP/(TP+FN))
print("F1: ", 2*TP/(2*TP+FP+FN))
print('\n')
avg_accuracy_improved = sum(all_accuracies_after_improving)/1+len(all_accuracies_after_improving)

print('Average accuracy is: ', avg_accuracy)
print('Average accuracy after improve is: ',avg_accuracy_improved)
print('My model improved succesfully' if avg_accuracy_improved > avg_accuracy else 'My model did not improve')



    




