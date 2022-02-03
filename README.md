This repository release the code and data of ShellFusion published in ICSE2022.
ShellFusion is an approach proposed to generate answers for shell programming tasks by fusing knowledge mined from three information sources, including 1) Q&A posts from Stack Overflow, Ask Ubuntu, Super User, and Unix & Linux; 2) Ubuntu Manual Pages (MPs): http://manpages.ubuntu.com; and 3) TLDR tutorials: https://github.com/tldr-pages/tldr.

The code are included in the three folders:
1. LuceneIndexerSearcher: contians the Java project for Lucene index and search.
2. offline: contains the offline data processing code of ShellFusion;
3. online: contains the online answer generation code of ShellFusion.

Since Github limits the maximum size of uploaded file to 25MB, we zipped the data in five 7z files:

1. queries.txt: includes the 434 experimental queries selected from Q&A shell-related questions. Each line is "Query Id (Q&A Community_Question Id) ===> Query ===> Preprocessed Query"
2. lucene_docs+index.7z: includes the docs file that contains the 537,129 shell-related questions, and the Lucene index built for the docs.
3. word_idf+word2vec.7z: includes the word IDF vocabulary and the word2vec model.
4. mpcmd_info.7z: includes the knowledge about shell commands extracted from Ubuntu MPs and TLDR tutorials.
5. QAPairs_det.7z: includes the shell commands with options detected from Q&A posts. Note that the QAPairs_det.json" file contains only the entire set of the top-N (=1,000) similar questions retrieved using Lucene search.
6. top-N+top-n.7z: includes the top-N (=1,000) similar questions retrieved using Lucene search and the top-n (=50) semantically similar questions retrieved using the word embedding approach for the 434 queries.