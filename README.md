# brown_clustering

Brown Clustering Implementation (Python)
- Kushagra Verma

-------------------------------------------------------------------
Directory Structure:
- brown/ = downloaded corpus from the homework link
- normalized/ = modified corpus after cleaning the corpus and inserting START, STOP and UNK symbols
Column 1 is rank, column 2 is name of token, column 3 is its frequency. All tokens whose frequency has been less than or equal to 10, have been replaced with a collective symbol, UNK
- brown_clusterting.py = script file

- sorted_vocabs.txt = output for q2(a). It lists out the vocabulary in a frequency + alphanumerically sorted order.
- encoding_map.txt = script output for binary encoding of the vocabulary
- encoding_map_200.txt = script output for binary encoding of the vocabulary with k = 200 BUT with random values of quality
====================================================================


-------------------------------------------------------------------
Steps to compile the project
- python brown_clustering.py, or
- python brown_clustering.py -k <number of clusters, K>
- python brown_clustering.py -k <number of clusters, K> fast

Example:
python brown_clustering.py
    - runs for default k=200 clusters
python brown_clustering.py -k 20
    - runs for k=20 clusters
python brown_clustering.py -k 200
    - runs for k=200 clusters
python brown_clustering.py -k 200 fast
    - runs for k=200 clusters BUT uses random values for quality.
    - This code compiles inside 5-6 mintues and generates a binary encoding for vocabulary as per the cluster tree.

** Please use the 'fast' flag for getting fast results to check program end-to-end correctness.
In this mode, the script uses random values for quality,
** but completes the program inside 5-6 minutes and generates a corresponding binary map

====================================================================



-------------------------------------------------------------------
High level pseudocode

- corpus cleaning step: clean the corpus and insert UNK, START, STOP symbols
- preprocessing step:
    - calculate token counts
    - rank tokens as per frequency
        - store result in sorted_vocabs.txt
    - calcualate bigram table
        - a 2D map of tokens
        - For example, self.bigram_table[apple][the] gives the tatal count of occurreces where 'the' came before 'apple'
    - start brown clustering
        - add initial k clusters of top k words
        - loop through k+1 ranked word to end of vocabulary
            - store quality of already calculated cluster pairs (i,j) to save time
        - merge remaining clusters to create the final cluster tree


====================================================================

-------------------------------------------------------------------
High Level Code Overview
Following are the set of classes. Please refer to inline code documentation for details

Pre-processors
    - NormalizeCorpus
        this class helps to:
        - clean the brown corpus
        - saves the corpus to normalized directory, after
            - cleaning tags
            - adding START, STOP, UNK
        - create a sorted_vocabs.txt having the sorted list of tokens
    - CorpusAnalyser
        this class helps to:
        - create a count map of all tokens
        - create a bigram table for all pairs of takens in the corpus
            - self.bigram_table[apple][the] gives the tatal count of occurreces where 'the' came before 'apple'
    - VocabParser
        - parse sorted_vocab to reconstruct token_frequency map

Data Structures
    - ClusterNode
        - this class models a cluster
        - this class object, cluster node is then used by the class defined below, cluster tree.
        - the implementation is similar to a binary tree,
            but each node contains a list of information defined in the constructor below
    - ClusterTre
        - this class models the binary cluster tree
        - nodes of the tree are cluster node objects
        - the tree also has helper methods to print binary encodings

Implementation
    - BrownClustering
        - class having the driver code to run the brown clustering algorithm
        - its initializes all the other class instances and pre computes values for quality optimization
        - algorithm outline,
            run_algorithm()
                self.create_top_k_clusters()        # step1: create k top clusters
                self.loop_through_vocab()           # step2: loop from k+1...V
                self.merge_remaining()              # step3: merge remaining clusters


    - rough pseudocode:
        # brown algorithm
        m = 200
        clusters = create m clusters
        for i = m+1 ... V:
            cm1 = create cluster c(m+1)
            append_clusters(clusters, cm1)

            max_quality = -inf
            max_cluster_tuple_index = (-1, -1)
            for i in m+1:
                for j in m+1:
                    cluster1 = f (i)
                    cluster2 = f (j)
                    merged_cluster = merge(cluster1, cluster2)
                    quality_current = calculate_quality(i, j, clusters)
                    if quality_current > quality_max:
                        quality_max = quality_current
                        max_cluster_tuple_index = (i, j)

            merge_cluster(i, j, clusters)

        for i: 0 -> m-1:
            final_merge(clusters, i)

        print_clusters_encoding(clusters)

====================================================================
====================================================================
