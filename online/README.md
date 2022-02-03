
This folder contains the online answer generation code of ShellFusion.
For the two-phase question retrieval method, we implement the Lucene indexer and searcher using a Java project "LuceneIndexerSearcher".

Given a query, please run the online steps as follows:
1. Preprocess the query using "query_preprocesser.py";
2. Retrieve the top-N similar questions using Lucene search engine, i.e., the "Searcher.java" in the Java project "LuceneIndexerSearcher";
3. Retrieve the top-n similar questions using a word embedding-based method, i.e., "SimQ_retriever.py";
4. Generate answers using "ShellFusion.py".
