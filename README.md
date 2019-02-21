# cross-arch-instr-model.github.io

Thank you for looking at our work!
The programs included here were created for the following paper:

"A Cross-Architecture Instruction Embedding Model for Natural Language Processing-Inspired Binary Code Analysis"

Kimberly Redmond, Lannan Luo, and Qiang Zeng

The NDSS Workshop on Binary Analysis Research (BAR), 2019.

############################

The trained cross-architecture instruction embedding model used in our paper are included in the output/ directory.

Our embeddings were trained on the model Bivec, which is based on Word2Vec.
You may find it here:

https://github.com/lmthang/bivec

Our training command used the following settings:

./bivec -src-train data/train.x86  -tgt-train data/train.arm -src-lang x86 -tgt-lang arm -output output/out -size 200 -window 5 -threads 4 -binary 0 -iter 10 -eval 1 -negative 30 -min-count 0 -sample 1e-5 -bi-weight 4

############################

ABOUT THESE PROGRAMS

All file paths and instruction selections are hard-coded into these programs. For your
convenience, they are listed in variables near the top; feel free to modify them for your use.

./senvec.py

Returns ROC plots and AUC scores for cross-architecture basic block similarity tests.
Basic block embeddings are calculated by:
	summing instruction embeddings within a block

Similarity is computed using Cosine similarity

./tsne2.py

Returns 2 t-SNE figures with different displays:
	1) an unlabeled figure displaying all instructions in one vector space
	2) a labeled figure displaying selected instructions in one vector space

./instr_sim.py

Returns 2 ROC plots and AUC scores for instruction-level similarity tests.
Instructions are evaluated in pairs, in 2 ways:
	1) mono-architecture
	2) cross-architecture

The similarity metric used is cosine similarity.

./query.py

Returns a list of the top-5 most similar instructions, given an instruction.
Each instruction returns the top 6 instructions from its own architecture
(#1 is itself), and the top 5 instructions from the other architecture,
according to cosine similarity.
