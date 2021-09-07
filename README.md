This project releases the source code and dataset of ShellFusion. 
ShellFusion is an approach proposed to generate answers for shell programming tasks by fusing knowledge mined from Q&A posts, Ubuntu Manual Pages (MPs), and TLDR tutorials.

The dataset includes:
1. queries.txt: This .txt file contains the 434 experimental queries selected from Q&A shell-related questions.
                Each line is "Query Id ===> Query ===> Preprocessed Query"
2. queries_userstudy.txt: This .txt file contains 10 queries used for a user study.
3. lucene_docs.txt: This .txt file contains the question documents created for 537,129 Q&A question repository.
4. lucene_index: This folder contains the index built for the question documents using Lucene.
5. w2v.kv: This .kv file stores the word2vec model.
6. token_idf.dump: This .dump file stores the word IDF vocabulary.
7. mpcmd_info.json: This .json file contains the knowledge of shell commands extracted from MPs and TLDR.
8. QAPairs_det.json: This .json file contains the commands with options detected from Q&A posts.
9. lucene_topN: This folder contains the top-N (=10,000) similar questions retrieved for 434 queries using Lucene search.
10. embed_topn: This folder contains the top-n (=50) similar questions retrieved for 434 queries using a word embedding approach.

NOTE: Since GitHub limits the maximum size of an uploaded file to 25MB, we zip the data in five .7z files
For other large amount of data, we will try to release using OneDrive later.

The source code includes:
1. ShellFusion: This Python project contains the main source code of ShellFusion.
2. LuceneIndexerSearcher: This Java projects contains the source code used for Lucene indexing and search.

A simple way to test ShellFusion with an example query:
Step 1: Download the source code and dataset.
Step 2: Open the ShellFusion project in an IDE, e.g., PyCharm.
Step 3: Unzip the dataset to a local directory, e.g., D:\\your_test_dir.
Step 4: Replace the "experiment_dir" in conf/conf.json by "D:\\your_test_dir".
Step 5: Replace the paths of the data files and dirs in "online/answer_generator.py".
Step 6: Run "online/answer_generator.py" for an example query, as listed in the .py file.
