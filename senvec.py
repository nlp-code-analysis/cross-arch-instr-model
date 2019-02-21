#!/usr/bin/python3

"""
    Sentence vector program.
    Takes word embeddings and raw instruction file.
    Stores word embeddings in dictionary.
    Scans raw instruction file and queries each embedding.
    Embeddings are summed per basic block.
    Basic blocks are compared across architecture languages.
"""

import math
import numpy
from numpy import linalg
from scipy import spatial
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


###      CHANGE FILES HERE      ###

# files containing word + embedding
param_file1a = "output/out.arm"
param_file1x = "output/out.x86"

# files containing test set of instructions
test_arm = "data/test.arm"
test_x86 = "data/test.x86"

# ground truth labels for test set
test_labels = "data/test_labels.txt"

# dimension size
dim = 200


# test set raw-text basic blocks
armtest = []
x86test = []

# dictionary for each trained lang/model
armdict1 = {}
x86dict1 = {}


"""
    Store the raw instructions from test files.
    Each line is a basic block.
"""
with open(test_arm) as f:
    for line in f:
        armtest.append(line.split())

with open(test_x86) as f:
    for line in f:
        x86test.append(line.split())


"""
    Each model will have its words + embeddings
    stored into a dictionary.
"""

def read_file(fname, dictionary, dim):
    with open(fname) as f:
        for line in f:

            # separate instructions and features
            items = line.split(" ")

            # remove header row
            if len(items) == 2:
                continue

            if "</s>" in items:
                continue

            if "<unk>" in items:
                continue

            if "\n" in items:
                endline = int(dim)+1
                del items[endline]

            instr = items[0]
            emb = numpy.array([float(feat) for feat in items[1:]])

            # turn corrupted read-in values to 0
            if numpy.any(numpy.isnan(emb)):
                print("Corrupted!: %s" % instr)
                emb = numpy.nan_to_num(emb)

            dictionary[instr] = emb


# Read in embeddings from each architecture file
print("Reading in embedding files...")
read_file(param_file1a, armdict1, dim)
read_file(param_file1x, x86dict1, dim)


# create sentence embedding for each bb
# find an instruction, query the dict
# sum instr embeddings to form a bb
def calculateSentenceVectors(testset, dim, dictry):
    finalout = []
    for bb in testset:
        count = 0 # number of embeddings found
        senvec = numpy.zeros([1, int(dim)])

        for word in bb:
            if word in dictry:
                wordvec = dictry[word]
                senvec += wordvec
                count += 1
            # else: unknown instructions are skipped

        if count > 0:

            # if averaging embeddings:
            # senvec /= float(count)
            finalout.append(senvec)

        else:
            print("Empty BB: no instructions found in dictionary.")

            # cannot use all zeros to compute cos similarity
            if numpy.all(senvec == 0.0):
                senvec.fill(0.1)

            finalout.append(senvec)

    return finalout

# calculate basic block vectors using hyperbolic tangent
def calculateTanhVectors(testset, dim, dictry):
    finalout = []
    for bb in testset:
        senvec = numpy.zeros([1, int(dim)])
        count = 0

        for word in bb:
            if word in dictry:

                if count % 2 == 1:
                    wordi1 = dictry[word]
                else:
                    wordi2 = dictry[word]

                if count > 0:
                    pair = wordi1 + wordi2
                    senvec += numpy.tanh(pair)
    
                count += 1
            
            # else: don't count words not in dictionary

        # cannot use all zeros to compute cos similarity
        if numpy.all(senvec == 0.0):
            senvec.fill(0.1)

        finalout.append(senvec)

    return finalout

print("Calculating sentence vectors...")
bbfinal_a1 = calculateSentenceVectors(armtest, dim, armdict1)
bbfinal_x1 = calculateSentenceVectors(x86test, dim, x86dict1)

print("Calculating tanh sentence vectors...")
bbfinal_at = calculateTanhVectors(armtest, dim, armdict1)
bbfinal_xt = calculateTanhVectors(x86test, dim, x86dict1)


###                 ###
#  Cosine Similarity  #
###                 ###

"""
    - Calculate cosine similarity between ARM/X86
        basic blocks
    - Plot ROC based on true labels
"""

true_labels = []

# ground truth labels
with open(test_labels) as f:
    for line in f:
        true_labels.append(line.split())
        
true_labels = [int(''.join((str(i) for i in a))) for a in true_labels]


# method to calculate cos similarity / ROC curve
def ROC(bbfinal_a, bbfinal_x):
    cos_sim = []
    for bb1, bb2 in zip(bbfinal_a, bbfinal_x):
        sim = 1 - spatial.distance.cosine(numpy.asarray(bb1), numpy.asarray(bb2))
        cos_sim.append(sim)

    return metrics.roc_curve(true_labels, cos_sim, pos_label=1)


# method to calculate Manhattan distance
def ROC_M(bbfinal_a, bbfinal_x):
    man_sim = []
    for bb1, bb2 in zip(bbfinal_a, bbfinal_x):
        sim = 1 - spatial.distance.cityblock(bb1, bb2)
        man_sim.append(sim)

    return metrics.roc_curve(true_labels, man_sim, pos_label=1)


# method to calculate Euclidean distance
def ROC_E(bbfinal_a, bbfinal_x):
    euc_sim = []
    for bb1, bb2 in zip(bbfinal_a, bbfinal_x):
        sim = 1 - numpy.linalg.norm(bb1-bb2)
        euc_sim.append(sim)

    return metrics.roc_curve(true_labels, euc_sim, pos_label=1)

print("Calculating similarity scores, ROC...")
fpr1, tpr1, thresholds1 = ROC(bbfinal_a1, bbfinal_x1)
fpr2, tpr2, thresholds2 = ROC_M(bbfinal_a1, bbfinal_x1)
fpr3, tpr3, thresholds3 = ROC_E(bbfinal_a1, bbfinal_x1)

fprt, tprt, thresholdst = ROC(bbfinal_at, bbfinal_xt)
fprt2, tprt2, thresholdst2 = ROC_M(bbfinal_at, bbfinal_xt)
fprt3, tprt3, thresholdst3 = ROC_E(bbfinal_at, bbfinal_xt)

auc1 = metrics.auc(fpr1, tpr1)
auc2 = metrics.auc(fpr2, tpr2)
auc3 = metrics.auc(fpr3, tpr3)

auct = metrics.auc(fprt, tprt)
auct2 = metrics.auc(fprt2, tprt2)
auct3 = metrics.auc(fprt3, tprt3)

print("AUC avg cos: %f" % auc1)
print("AUC avg man: %f" % auc2)
print("AUC avg euc: %f" % auc3)

print("AUC tanh cos: %f" % auct)
print("AUC tanh man: %f" % auct2)
print("AUC tanh euc: %f" % auct3)


plt.title('ARM-X86 Basic Block Similarity')
plt.plot(fpr1, tpr1, label="Cosine Similarity, AUC=%f" %auc1)
plt.plot(fpr2, tpr2, label="Manhattan Distance, AUC=%f" %auc2)
plt.plot(fpr3, tpr3, label="Euclidean Distance, AUC=%f" %auc3)

plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.title('ARM-X86 Basic Block Similarity (Tanh)')
plt.plot(fprt, tprt, label="Cosine Similarity, AUC=%f" %auct)
plt.plot(fpr2, tpr2, label="Manhattan Distance, AUC=%f" %auct2)
plt.plot(fpr3, tpr3, label="Euclidean Distance, AUC=%f" %auct3)

plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
