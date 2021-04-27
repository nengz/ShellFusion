This project release the code and data of ShellFusion.

The code are included in the three folders:
1. LuceneIndexerSearcher: contians the Java project for Lucene index and search.
2. offline: contains the offline data processing code of ShellFusion;
3. online: contains the online answer generation code of ShellFusion.

Since Github limits the maximum size of uploaded file to 25MB, we zipped the data in three 7z files:
1. lucene docs+index.7z: includes the docs file that contains all the 537,129 shell-related question documents, and the Lucene index built for the docs.
3. word idf+word2vec.7z: includes the word IDF vocabulary and the word2vec model.
4. mp+tldr cmds.7z: includes the knowledge about shell commands extracted from Ubuntu MPs and TLDR tutorials.
