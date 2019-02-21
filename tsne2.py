#!/usr/bin/python

"""
    t-SNE to create a visualization of x86, arm embeddings.
    Output: 2 t-SNE plots
        1) unlabeled plot displaying all instructions
        2) labeled plot displaying selected instructions
"""

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import math
import numpy
from numpy import dot
from numpy.linalg import norm
from adjustText import adjust_text


##                          ##
#            data            #
##                          ##


###      CHANGE FILES HERE      ###

DATA_PATH_ARM = 'output/out.arm'
DATA_PATH_X86 = 'output/out.x86'
VECTOR_SIZE = 200


# select instructions to plot only a few
armcodes = ['ADD~R1,R0,R7', 'SUB~SP,SP,0',
            'LDR~R0,[R5+0]', 
            'CMP~R8,0', 'POP~{R4,LR}',
            'B~<TAG>', 'BL~FOO']
x86codes = ['ADDQ~R13,RBX', 'SUBQ~RSP,0',
            'MOVQ~RDI,[R12+0]', 
            'CMPL~R13D,0', 'POPQ~RBP',
            'JMP~<TAG>', 'CALLQ~FOO']
instr_ct = 7


all_arm_codes = []
all_x86_codes = []

armdict = {}
x86dict = {}

arm_embedding_matrix = numpy.empty([instr_ct, VECTOR_SIZE])
x86_embedding_matrix = numpy.empty([instr_ct, VECTOR_SIZE])


##                          ##
#            code            #
##                          ##

# construct ARM dictionary
print("Constructing arm dictionary...")
with open(DATA_PATH_ARM) as f:
    for line in f:
        items = line.split(" ")

        # remove header row
        if len(items) == 2:
            dim1 = items[1]
            continue

        if "</s>" in items:
            continue

        if "\n" in items:
            endline = int(dim1)+1
            del items[endline]

        instr = items[0]
        emb = numpy.array([float(feat) for feat in items[1:]])

        armdict[instr] = emb

# construct X86 dictionary
print("Constructing x86 dictionary...")
with open(DATA_PATH_X86) as f:
    for line in f:
        items = line.split(" ")

        # remove header row
        if len(items) == 2:
            dim1 = items[1]
            continue

        if "</s>" in items:
            continue

        if "\n" in items:
            endline = int(dim1)+1
            del items[endline]

        instr = items[0]
        emb = numpy.array([float(feat) for feat in items[1:]])

        x86dict[instr] = emb


##                    ##
#     TSNE and PLOT    #
##                    ##

print("Compiling all embeddings together...")

arm_matrix = numpy.empty([len(armdict), VECTOR_SIZE])
x86_matrix = numpy.empty([len(x86dict), VECTOR_SIZE])
arm_index = 0
x86_index = 0

for embedding in armdict.values():
    arm_matrix[arm_index] = embedding
    arm_index += 1

for embedding in x86dict.values():
    x86_matrix[x86_index] = embedding
    x86_index += 1

# compile all embeddings together
final_len = len(armdict) + len(x86dict)
final_matrix = numpy.empty([final_len, VECTOR_SIZE])
index = 0

for elem in arm_matrix:
    final_matrix[index] = elem
    index += 1

for elem in x86_matrix:
    final_matrix[index] = elem
    index += 1

### final_matrix contains ALL embeddings

# check length vs matrix size
print("Final len: %d Matrix size: %d" % (final_len, index))

# dimension reduction
print("Running PCA...")
pca = PCA(n_components = 50)
new_final_matrix = pca.fit_transform(final_matrix)

print("Running TSNE...")
tsne_matrix = TSNE(n_components=2).fit_transform(new_final_matrix)


# Plot
matplotlib.rcParams.update({'font.size': 7})

# We do not want to print out all possible instructions
# when plotting the queried instructions.
# This creates a new matrix for our small sample.
plot_arm_matrix = numpy.zeros([instr_ct,2])
plot_x86_matrix = numpy.zeros([instr_ct,2])
plot_all_matrix = numpy.zeros([(instr_ct*2),2])
plot_arm_index = 0
plot_x86_index = 0
plot_all_index = 0

armlabels = []
x86labels = []
alllabels = []

# compile all instruction names
instr_list = []

for arminstr in armdict.keys():
    instr_list.append(arminstr)

for x86instr in x86dict.keys():
    instr_list.append(x86instr)


###     TSNE - All Instructions     ###

print("Plotting ALL instructions:")
print("Setting scatter plot...")

final_arm_matrix = numpy.zeros([len(armdict), 2])
final_x86_matrix = numpy.zeros([len(x86dict), 2])
armcount = 0
x86count = 0

for x, y in zip(tsne_matrix[:,0], tsne_matrix[:,1]):
    if armcount > (len(armdict) - 1):
        final_x86_matrix[x86count,] = x,y
        x86count += 1
    else:
        final_arm_matrix[armcount,] = x,y
        armcount += 1

plt.scatter(final_arm_matrix[:,0], final_arm_matrix[:,1], s=50, marker="o", c='blue', alpha=0.1)
plt.scatter(final_x86_matrix[:,0], final_x86_matrix[:,1], s=50, marker="^", c='red', alpha=0.1)

plt.xticks([])
plt.yticks([])
plt.show()

###     TSNE - Selected Instructions    ###

for key, x, y in zip(instr_list, tsne_matrix[:,0], tsne_matrix[:,1]):
    if key in armcodes:
        plot_arm_matrix[int(plot_arm_index),] = x,y
        armlabels.append(key)
        plot_arm_index += 1

    if key in x86codes:
        plot_x86_matrix[int(plot_x86_index),] = x,y
        x86labels.append(key)
        plot_x86_index += 1

# Plot selected instructions only
print("Plotting selected instructions only:")
print("Setting scatter plot...")

for label, x, y in zip(armlabels, plot_arm_matrix[:,0], plot_arm_matrix[:,1]):
    plt.annotate(
        label,
        size = 12,
        xy = (x, y),
        ha = 'left',
        va = 'top')

for label, x, y in zip(x86labels, plot_x86_matrix[:,0], plot_x86_matrix[:,1]):
    plt.annotate(
        label,
        size = 12,
        xy = (x, y),
        ha = 'right',
        va = 'bottom')

plt.scatter(plot_arm_matrix[:,0], plot_arm_matrix[:,1], s=50, marker="o", c='blue', alpha=0.8)
plt.scatter(plot_x86_matrix[:,0], plot_x86_matrix[:,1], s=50, marker="^", c='red', alpha=0.8)


plt.xticks([])
plt.yticks([])
plt.show()

