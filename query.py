#!/usr/bin/python3

"""
    Query Program for Word Similarity.
    - Query the top-10 most similar instructions (according to
      cosine similarity) within and across architectures:
        ARM-ARM
        ARM-x86
        x86-x86
        x86-ARM
    - Return 5 most similar instructions and their sim scores.
"""

import math
import numpy
from scipy import spatial
from sklearn import metrics
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt


###      CHANGE FILES HERE      ###

# Files containing embeddings
arm_out_emb = "output/out.arm"
x86_out_emb = "output/out.x86"

# Dimension size
dim = 200

# Instructions to query
arm_query = ['ADD~SP,SP,0', 'SUB~SP,SP,0',
            'LDR~R0,[R5+0]', 'BL~FOO', 
            'LDRNE~R4,[SP+0]', 'ADD~R1,R0,R7',
            'BLT~<TAG>', 'BEQ~<TAG>', 'MOV~R0,R5',
            'MOV~R8,R2', 'ADD~R1,R0,R7', 'SUB~SP,SP,0',
            'LDR~R0,[R5+0]',  'MVN~R0,0', 'CMP~R8,0']
x86_query = ['ADDQ~RSP,0', 'SUBQ~RSP,0',
            'MOVQ~RDI,[R12+0]','CALLQ~FOO',
            'CMOVNEQ~RCX,RAX', 'ADDQ~RSI,R12',
            'JLE~<TAG>', 'JE~<TAG>', 'MOVL~EAX,R14D',
            'MOVQ~R13,RDX', 'ADDQ~R13,RBX', 'SUBQ~RSP,0',
            'MOVQ~RDI,[R12+0]', 'MOVL~EAX,-0', 'CMPL~R13D,0']
noinstr = 15


##                        ##
#   ARM/x86 Dictionaries   #
##                        ##
armdict = {}
x86dict = {}
armlist = []
x86list = []


# Method to read from embedding files
# Stored as [instruction]:embedding pairs
def read_file(fname, dictionary, dim, lst):
    with open(fname) as f:
        for line in f:
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

            # turn corrupted values to 0
            if numpy.any(numpy.isnan(emb)):
                print("Corrupted!: %s" % instr)
                #emb = numpy.nan_to_num(emb)

            dictionary[instr] = emb
            lst.append(instr)


##                          ##
#   Construct dictionaries   #
##                          ##
print("Constructing dictionaries from files...")
read_file(arm_out_emb, armdict, dim, armlist)
read_file(x86_out_emb, x86dict, dim, x86list)


##                               ##
#   Calculate similarity scores   #
##                               ##
print("Calculating similarity scores...")

arm = len(armlist)
x86 = len(x86list)

arm2arm = numpy.ndarray([noinstr, arm])
arm2x86 = numpy.ndarray([noinstr, x86])
x862x86 = numpy.ndarray([noinstr, x86])
x862arm = numpy.ndarray([noinstr, arm])

def cosinesim(instr1, instr2):
    emb1 = instr1.reshape(1,-1)
    emb2 = instr2.reshape(1,-1)
    sim = pairwise.cosine_similarity(emb1, emb2)
    return sim


# Calc similarities for each query instruction
row = 0
for instr in arm_query:
    emb = armdict[instr]
    index = 0
    for instr2 in armdict.values():
        arm2arm[row,index] = cosinesim(emb, instr2)
        index += 1

    index = 0
    for instr2 in x86dict.values():
        arm2x86[row,index] = cosinesim(emb, instr2)
        index += 1

    row += 1

row = 0
for instr in x86_query:
    emb = x86dict[instr]
    index = 0
    for instr2 in x86dict.values():
        x862x86[row,index] = cosinesim(emb, instr2)
        index += 1

    index = 0
    for instr2 in armdict.values():
        x862arm[row,index] = cosinesim(emb, instr2)
        index += 1

    row += 1


##                               ##
#   Top 5 Similar Instructions   #
##                               ##

print("Returning top 5 similar instructions...")

row = 0
for instr in arm_query:
    armrow = arm2arm[row]
    x86row = arm2x86[row]

    print("\tARM-ARM\t%s" % instr)
    for x in range(6):
        cos = numpy.max(armrow) # top cos value
        index = numpy.argmax(armrow) # index of that value
        word = armlist[index] # find word in dictionary

        print("%s\t\t\t%f" % (word, cos))
        armrow = numpy.delete(armrow, index)

    print("\n")

    print("\tARM-X86\t%s" % instr)
    for x in range(5):
        cos = numpy.max(x86row)
        index = numpy.argmax(x86row)
        word = x86list[index]

        print("%s\t\t\t%f" % (word, cos))
        x86row = numpy.delete(x86row, index)

    print("\n")
    
    row += 1

row = 0
for instr in x86_query:
    armrow = x862arm[row]
    x86row = x862x86[row]

    print("\tX86-X86\t%s" % instr)
    for x in range(6):
        cos = numpy.max(x86row)
        index = numpy.argmax(x86row)
        word = x86list[index]

        print("%s\t\t\t%f" % (word, cos))
        x86row = numpy.delete(x86row, index)

    print("\n")

    print("\tX86-ARM\t%s" % instr)
    for x in range(5):
        cos = numpy.max(armrow)
        index = numpy.argmax(armrow)
        word = armlist[index]

        print("%s\t\t\t%f" % (word, cos))
        armrow = numpy.delete(armrow, index)

    print("\n")

    row += 1
