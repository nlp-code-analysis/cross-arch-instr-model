#!/usr/bin/python3

"""
    Kim Redmond
    A program to evaluate cosine similarity
    between INSTRUCTIONS, not basic blocks.
    These instructions are randomly hand-chosen.
        - ARM-ARM (100)
        - x86-x86 (100)
        - ARM-x86 (50)
    Test accuracy by plotting ROC/AUC.
"""

import math
import numpy
from scipy import spatial
from sklearn import metrics
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt


###      CHANGE FILES HERE      ###

arm_emb = "output/out.arm"
x86_emb = "output/out.x86"
dim = 200


###
#  Instruction pairs to test...
###

# mono-architecture
# 50 similar, 50 dissimilar

arm1codes = ['ADD~R0,R5,R7', 'MOV~R1,0', 'CMP~R0,0', 'ORR~R0,R0,0', 'AND~R12,R6,0',
            'ADD~R0,R9,R7,LSL0', 'MOV~R0,R4', 'LDR~R0,[R0]', 'ORRS~R0,R5,R7', 'ADDS~R1,R1,0',
            'SBCS~R2,R2,0', 'SUBS~R2,R1,0', 'MOVEQ~R5,0', 'LDRB~R1,[R6+R7]', 'MVN~R12,0',
            'MOVNE~R2,R4', 'STM~R2,{R0,R1,R5}', 'ADC~R9,R5,R4', 'LSL~R1,R3,0', 'EOR~R11,R6,R1',
            'STMIB~SP,{R0,R7}', 'LDRLO~R1,[SP+0]', 'LDM~R7,{R3,R7}', 'RSBS~R3,R3,0', 'ORREQ~R6,R11,0',
            'ANDNE~R3,R3,0', 'MOVGE~R7,0', 'RSCS~R1,R10,0', 'ADDEQ~R8,R8,0', 'TST~R0,0',
            'LDRBNE~R7,[R9+R7]', 'STR~R6,[R0]', 'STRB~R0,[R2-0]', 'UMULL~R1,R2,R0,R3', 'PUSH~{R4,R5,R6,R7,R8,R9,R10,R11,LR}',
            'BIC~SP,SP,0', 'MLA~R10,R2,R4,R0', 'SBC~R3,R12,0', 'MOV~R2,R7', 'MUL~R1,R0,R2',
            'LDMIB~R7,{R2,R3}', 'MOVLE~R0,R1', 'ASR~R0,R0,0', 'LSLGE~R2,R3,R7', 'POP~{R4,R5,R6,R7,R8,R9,R10,R11,LR}',
            'MVNNE~R7,0', 'ASRNE~R6,R6,0', 'STMNE~R0,{R6,R8}', 'SUB~R0,R8,R0', 'LDREQ~R0,<TAG>',
            'ADD~R0,R5,R7', 'ADDS~R1,R1,0', 'ADD~R0,SP,0', 'ADDS~R4,R4,0', 'ADD~R9,R10,R10,LSL0',
            'ADD~R10,SP,0', 'ADD~R11,SP,0', 'ADD~R1,SP,0', 'ADD~R0,R0,R4', 'ADDS~R2,R11,R0',
            'BL~FOO', 'BEQ~<TAG>', 'MVN~R2,0', 'UMULL~R2,R3,R0,R1', 'ADC~R0,R5,R6',
            'LDR~R1,<TAG>', 'LDR~R1,[R5]', 'MOV~R9,R8', 'MOVNE~R0,0', 'B~<TAG>',
            'PUSH~{R4,R5,R6,R7,R8,LR}', 'RSCS~R3,R1,R4,ASR0', 'BHS~<TAG>', 'ORRSNE~R2,R2,R3', 'CMN~R0,0',
            'ORR~R8,R1,R8', 'ORR~R10,R3,R10', 'ORRS~R1,R0,R2', 'ADD~R6,SP,0', 'BLO~<TAG>',
            'MOV~R2,R5', 'MOV~R2,0', 'MOV~R0,R9', 'MOV~R6,R4', 'MOV~R0,R8',
            'STRH~R0,[SP+0]', 'CMPNE~R1,0', 'SUBS~R2,R4,0', 'MOVNE~R1,R4', 'ORR~R1,R1,0',
            'POP~{R4,R5,R6,LR}', 'UMULL~R3,R7,R0,R4', 'UMULL~R6,R5,R3,R4', 'CMPEQ~R6,0', 'MOV~R0,R10',
            'ASR~R11,R0,0', 'ORRNE~R6,R8,R6,LSL0', 'LDR~R2,[SP+0]', 'RSBS~R0,R5,0', 'BLS~<TAG>']

arm2codes = ['ADD~R6,R6,R10', 'MOV~R10,0', 'CMP~R4,0', 'ORR~R2,R2,0', 'AND~R1,R1,0',
            'ADD~R2,R1,R8,LSL0', 'MOV~R0,R9', 'LDR~R0,[SP+0]', 'ORRS~R0,R0,R1', 'ADDS~R4,R1,0',
            'SBCS~R0,R4,R1', 'SUBS~R2,R0,R4', 'MOVEQ~R0,0', 'LDRB~R5,[R4+R7+LSR0]', 'MVN~R2,0',
            'MOVNE~R7,0', 'STM~R10,{R0,R1}', 'ADC~R3,R10,0', 'LSL~R2,R7,0', 'EOR~R0,R5,0',
            'STMIB~R0,{R8,R9}', 'LDRLO~R1,[R12]', 'LDM~R1,{R0,R1}', 'RSBS~R4,R4,0', 'ORREQ~R9,R9,R5,LSL0',
            'ANDNE~R7,R4,R6,LSR0', 'MOVGE~R1,0', 'RSCS~R1,R2,0', 'ADDEQ~R2,R0,0', 'TST~R3,0',
            'LDRBNE~R3,[R0]', 'STR~R4,[SP+0]', 'STRB~R7,[R0],0', 'UMULL~R8,R0,R2,R6', 'PUSH~{R4,R5,R6,R7,R8,R9,R10,LR}',
            'BIC~R0,R0,0', 'MLA~R2,R3,R8,R1', 'SBC~R3,R5,R3', 'MOV~PC,R2', 'MUL~R3,R0,R2',
            'LDMIB~R7,{R0,R1}', 'MOVLE~R1,R0', 'ASR~R3,R2,0', 'LSLGE~R4,R0,R3', 'POP~{R4,LR}',
            'MVNNE~R0,0', 'ASRNE~R0,R5,0', 'STMNE~R1,{R3,R12}', 'SUB~SP,R11,0', 'LDREQ~R0,[R10+0]',
            'SUBS~R2,R0,R4', 'SUB~R0,SP,R0', 'SUBS~R4,R2,R0', 'SUBS~R0,R0,0', 'SUBS~R7,R10,R7',
            'BNE~<TAG>', 'LSR~R0,R0,0', 'MOV~R8,R0', 'BLT~<TAG>', 'B~<TAG>',
            'ADC~R4,R0,0', 'STM~SP,{R0,R4}', 'BL~FOO', 'LDRB~R0,[R5+0]!', 'ORRS~R0,R0,R1',
            'TST~R11,0', 'BHI~<TAG>', 'ADD~SP,SP,0', 'LDRB~R6,[R4]', 'LSL~R2,R1,0',
            'POP~{R4,R5,R6,R7,R8,R9,R10,R11,LR}', 'CMP~R0,0', 'ADD~R1,R0,0', 'ANDS~R5,R8,0', 'AND~R6,R2,0',
            'AND~R2,R0,0', 'AND~R0,R0,0', 'ASR~R1,R0,0', 'SUB~R0,R6,0', 'BIC~R1,R7,0',
            'BGE~<TAG>', 'LDMIB~R6,{R2,R3}', 'SUBS~R7,R2,R0', 'AND~R0,R0,0', 'AND~R4,R1,0',
            'BL~FOO', 'STR~R6,[SP+0]', 'ADDS~R10,R10,0', 'AND~R2,R2,R1', 'CMN~R9,-0',
            'PUSH~{R11,LR}', 'LDM~R2,{R3,R6}', 'LDRB~R6,[R1+0]', 'ADC~R10,R10,0', 'BNE~<TAG>',
            'SUB~SP,SP,0', 'TST~R0,0', 'BNE~<TAG>', 'EOR~R0,R5,0', 'MLA~R3,R10,R6,R2']

x861codes = ['ADDQ~RSP,0', 'MOVQ~RDI,RBX', 'CMPQ~RDX,0', 'XORL~EAX,EAX', 'ANDQ~R15,-0',
            'ADDQ~R14,RAX', 'MOVQ~RDX,RCX', 'LEAQ~RAX,[RBX+0]', 'XORL~EDI,EDI', 'ADDL~EAX,-0',
            'SUBQ~RSP,0', 'SUBL~EAX,R15D', 'MOVL~EAX,EBX', 'CMPL~ESI,0', 'SHLL~EBX,0',
            'MOVUPS~[RAX+0],XMM0', 'TESTQ~RCX,RCX', 'ADCQ~RBX,0', 'ANDB~AL,0', 'CMPB~AL,0',
            'ORL~ECX,EAX', 'TESTB~DL,0', 'MOVZBL~EDX,[RAX+0]', 'SHRL~ESI,0', 'ORL~ESI,0',
            'IDIVL~EDI', 'CMOVNEQ~R13,R15', 'NEGQ~R10', 'SETE~DL', 'ORQ~RDX,RDI',
            'NOTL~ECX', 'MOVSLQ~R8,[R13+0]', 'CMOVBL~EAX,ESI', 'NOTQ~RDX', 'CMOVAL~R11D,EDI',
            'PSHUFD~XMM0,XMM1,0', 'PUNPCKLDQ~XMM0,[RIP+<TAG>]', 'SUBPD~XMM0,[RIP+<TAG>]', 'MOVAPD~[RSP],XMM1', 'XORPS~XMM0,XMM0',
            'MOVAPS~[RSP+0],XMM0', 'NOTB~BL', 'CALLQ~*RAX', 'MOVABSQ~RCX,0', 'POPQ~RBX',
            'MOVDQU~[RDI,RDX,0-0],XMM2', 'PSHUFLW~XMM3,XMM3,0', 'PSRAD~XMM2,0', 'PUNPCKLBW~XMM2,XMM2', 'PADDQ~XMM3,XMM1',
            'ADDQ~RSP,0', 'ADDQ~R12,0', 'ADDQ~RAX,-0', 'PADDQ~XMM0,XMM2', 'ADDQ~R13,0',
            'NOTQ~RCX', 'CLTQ', 'NEGL~R9D', 'SETE~R15B', 'SHRB~BL,0',
            'JMP~<TAG>', 'CALLQ~FOO', 'RETQ', 'JAE~<TAG>', 'JNE~<TAG>',
            'MOVB~AL,0', 'MOVL~EAX,0', 'MOVL~ESI,0', 'MOVZBL~EAX,BL', 'MOVL~EDI,<STR>',
            'XORL~ECX,ECX', 'XORL~EBX,EBX', 'XORL~EAX,EAX', 'XORL~R8D,R8D', 'XORL~EDI,EDI',
            'CMPL~R14D,0', 'CMPQ~RDX,RBX', 'CMPL~EAX,-0', 'CMPQ~R14,-0', 'CMPQ~[R12+0],0',
            'MOVW~<TAG>+[RIP+0],0', 'MOVQ~RCX,RBX', 'MOVZBL~EAX,[RIP+<TAG>]', 'MOVQ~RDI,R12', 'MOVQ~RBX,RDI',
            'LEAQ~RSI,[RBX+0]', 'LEAQ~RSI,[RCX+0]', 'LEAQ~RDI,[RSP+0]', 'LEAL~EDI,[RDX,RSI]', 'LEAL~EBP,[RSI-0]',
            'PUSHQ~RBP', 'PUSHQ~R15', 'PUSHQ~R12', 'PUSHQ~RBX', 'PUSHQ~R14',
            'JMPQ~*[RAX*0+<TAG>]', 'CALLQ~*[RIP+<TAG>]', 'CMOVGL~R8D,EDX', 'IMULL~EDI,EDX,0', 'SBBL~EBP,EBP']

x862codes = ['ADDQ~R14,R12', 'MOVQ~R14,RSP', 'CMPQ~[RIP+<TAG>],RAX', 'XORL~R14D,R14D', 'ANDQ~RAX,R15',
            'ADDQ~RDX,-0', 'MOVQ~RAX,[R8+0]', 'LEAQ~R14,[R8+0]', 'XORL~EBX,EBX', 'ANDQ~RBP,-0',
            'SUBQ~RBX,RAX', 'SUBL~ESI,EBP', 'MOVL~EDX,[RBX+0]', 'CMPL~R15D,0', 'SHLL~EDX,CL',
            'MOVUPS~XMM0,[RIP+<TAG>]', 'TESTQ~R8,R8', 'ADCQ~R8,0', 'ANDB~[RBX+0],-0', 'CMPB~[RSP+0],0',
            'ORL~EAX,R8D', 'TESTB~[RSI+0],0', 'MOVZBL~EDX,R13B', 'SHRL~ECX,0', 'ORL~EAX,0',
            'IDIVL~R15D', 'CMOVNEQ~RAX,R10', 'NEGQ~RDX', 'SETE~CL', 'ORQ~RAX,R15',
            'NOTL~EDX', 'MOVSLQ~RSI,EBP', 'CMOVBL~EAX,ECX', 'NOTQ~RAX', 'CMOVAL~EAX,ECX',
            'PSHUFD~XMM2,XMM1,0', 'PUNPCKLDQ~XMM2,[RIP+<TAG>]', 'SUBPD~XMM2,[RIP+<TAG>]', 'MOVAPD~XMM5,XMM1', 'XORPS~XMM1,XMM1',
            'MOVAPS~XMM0,[RIP+<TAG>]', 'NOTB~[R15]', 'CALLQ~FOO', 'MOVABSQ~RAX,0', 'POPQ~R13',
            'MOVDQU~XMM0,[R12]', 'PSHUFLW~XMM0,XMM0,0', 'PSRAD~XMM0,0', 'PUNPCKLBW~XMM5,<TAG>', 'PADDQ~XMM2,XMM10',
            'SUBL~ESI,EBP', 'SUBL~EAX,R15D', 'SUBL~ESI,[RSP+0]', 'SUBQ~RSP,0', 'SUBL~EAX,R12D',
            'ADDL~R12D,-0', 'ORL~EBP,R12D', 'TESTQ~RAX,RAX', 'CMPB~[R15,RBX],0', 'ORB~DL,R8B',
            'ANDL~EAX,0', 'XORL~ESI,ESI', 'POPQ~R15', 'MOVB~R15B,0', 'MOVZBL~ECX,[R14+0]',
            'SHLQ~R8,0', 'POPQ~RBX', 'POPQ~R14', 'JE~<TAG>', 'JA~<TAG>',
            'TESTB~[RBX+0],0', 'TESTL~EAX,EAX', 'TESTQ~RDI,RDI', 'ANDL~R11D,0', 'ANDL~R9D,0',
            'ADDQ~R14,RBX', 'XORL~ESI,ESI', 'JAE~<TAG>', 'JMP~<TAG>', 'TESTQ~RAX,RAX',
            'ANDL~ECX,-0', 'RETQ', 'CMPB~[R12],0', 'POPQ~RBX', 'TESTL~EAX,EAX',
            'CMPQ~RBX,0', 'TESTB~BPL,0', 'SETNE~AL', 'JMP~<TAG>', 'CMPQ~RBP,0',
            'POPQ~RBX', 'POPQ~RBP', 'POPQ~R15', 'POPQ~R12', 'XORL~EAX,EAX',
            'SHRQ~R8,0', 'MOVUPS~[RCX],XMM0', 'SUBQ~RBP,RDI', 'JB~<TAG>', 'CALLQ~FOO']
true_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
num = 100

# cross-architecture instructions
# 25 similar, 25 dissimilar
arm_cross = ['ADD~R0,R5,0', 'ADD~R6,R0,0', 'ADC~R7,R1,0', 'ADD~SP,SP,0', 'ADDS~R0,R2,R0',
             'ORR~R0,R0,0', 'ORR~R9,R9,0', 'AND~R0,R9,0', 'AND~R2,R2,0', 'ORRS~R3,R1,R2',
             'SUBS~R0,R6,0', 'SUB~R3,R3,R8,LSL0', 'SUBS~R0,R7,R10', 'SUB~SP,SP,0', 'SUBS~R2,R10,0',
             'CMP~R0,0', 'MOV~R0,R5', 'MOV~R6,0', 'LDR~R0,[R8]', 'LDR~R0,[R10+0]',
             'BL~FOO', 'BNE~<TAG>', 'EOR~R1,R4,R9,ASR0', 'B~<TAG>', 'STR~R8,[R4]',

             'ADDNE~R0,R12,0', 'ADC~R6,R6,0', 'ADC~R1,R1,0', 'ADDS~R9,R9,0', 'ADDS~R0,R0,R7',
             'ORR~R1,R1,R0,LSR0', 'ORRS~R0,R6,R7', 'AND~R0,R4,0', 'AND~R0,R3,R2', 'ORRS~R0,R0,R11',
             'SUB~R11,R0,R3', 'SUBS~R2,R0,0', 'SUBS~R6,R0,R2', 'LDR~R2,[SP+0]', 'LDR~R0,[R6]',
             'POP~{R4,LR}', 'PUSH~{R4,R5,R6,R7,R8,R9,R10,R11,LR}', 'MOV~R9,R2', 'MOVEQ~R2,R1', 'CMP~R5,0',
             'TST~R0,0', 'BL~FOO', 'BEQ~<TAG>', 'SBC~R7,R7,0', 'SMULL~R6,R5,R7,LR']

x86_cross = ['ADDQ~RAX,0', 'ADDQ~RBX,RAX', 'ADDL~ECX,-0', 'ADDQ~RSP,0', 'ADDL~EBP,0',
             'XORL~EBP,EBP', 'ORQ~RCX,RDX', 'ANDL~EDX,0', 'TESTQ~RCX,RCX', 'ORQ~RSI,RBX',
             'SUBQ~RAX,[RSP+0]', 'SUBQ~R12,RAX', 'SUBL~ESI,EBX', 'SUBQ~RSP,0', 'SUBL~ESI,R12D',
             'CMPQ~[R14],0', 'MOVQ~RBX,RAX', 'MOVQ~RDI,RAX', 'LEAQ~RBX,[R14,RBP]', 'LEAQ~RSI,[RSP+0]',
             'CALLQ~FOO', 'JNE~<TAG>', 'XORL~EDX,EDX', 'JMP~<TAG>', 'MOVQ~RAX,[RBX]',

             'CMOVLL~ECX,EAX', 'PUSHQ~RBX', 'PSUBQ~XMM0,XMM1', 'XORL~EAX,EAX', 'SUBQ~RSP,0',
             'POPQ~RBP', 'MOVQ~RDI,R13', 'TESTB~AL,AL', 'CMOVNEQ~RBX,RCX', 'IMULQ~RBX,R8',
             'CMPQ~[RSP+0],0', 'CMPL~EAX,0', 'MOVZWL~EAX,[R13+0]', 'PUSHQ~R15', 'JS~<TAG>',
             'BTQ~R14,RAX', 'JA~<TAG>', 'DIVB~[RSP+0]', 'JNE~<TAG>', 'MOVSLQ~R13,EBP',
             'SUBL~EAX,[RIP+<TAG>]', 'LEAL~EAX,[R14,RBP]', 'RETQ', 'MOVQ~RDI,RAX', 'ADDQ~RBX,RBP']

cross_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

armdict = {}
x86dict = {}


# Method to read from embedding files
# Stored as [instruction]:embedding pairs
def read_file(fname, dictionary, dim):
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
                emb = numpy.nan_to_num(emb)

            dictionary[instr] = emb


##                          ##
#   Construct dictionaries   #
##                          ##
print("Constructing dictionaries from files...")
read_file(arm_emb, armdict, dim)
read_file(x86_emb, x86dict, dim)


##                           ##
#   Instr Cosine Similarity   #
##                           ##
print("Calculating cosine similarities...")

def cos_sim(out_list, dictry, codes1, codes2):
    for x, y in zip(codes1, codes2):
        num1 = dictry[x]
        num2 = dictry[y]

        emb1 = num1.reshape(1, -1)
        emb2 = num2.reshape(1, -1)

        sim = pairwise.cosine_similarity(emb1, emb2) # returns array

        print("%s %s     %f" % (x, y, sim))
        out_list.append(sim[0,0])

arm_cos = []
cos_sim(arm_cos, armdict, arm1codes, arm2codes)

x86_cos = []
cos_sim(x86_cos, x86dict, x861codes, x862codes)

cross_cos = []
for x,y in zip(arm_cross, x86_cross):
    num1 = armdict[x]
    num2 = x86dict[y]

    emb1 = num1.reshape(1,-1)
    emb2 = num2.reshape(1,-1)
    sim = pairwise.cosine_similarity(emb1, emb2) # returns array

    print("%s %s       %f" % (x, y, sim))
    cross_cos.append(sim[0,0])

# ROC, AUC
fpra, tpra, thresholdsa = metrics.roc_curve(true_labels, arm_cos, pos_label=1)
fprx, tprx, thresholdsx = metrics.roc_curve(true_labels, x86_cos, pos_label=1)
fprc, tprc, thresholdsc = metrics.roc_curve(cross_labels, cross_cos, pos_label=1)

auca = metrics.auc(fpra, tpra)
aucx = metrics.auc(fprx, tprx)
aucc = metrics.auc(fprc, tprc)

plt.title("Mono-Architecture Instruction-Level Similarity")
plt.plot(fpra, tpra, label="ARM Instructions, AUC=%f" % auca)
plt.plot(fprx, tprx, label="X86 Instructions, AUC=%f" % aucx)

plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

plt.title("Cross-Architecture Instruction-Level Similarity")
plt.plot(fprc, tprc, label="ARM-X86 Instructions, AUC=%f" % aucc)

plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
