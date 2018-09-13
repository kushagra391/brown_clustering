
import math
import pprint
import os
import random
import re
import sys
import time
from collections import defaultdict

tab_delim = "\t"
space_delim = " "
hash_delim = "/"
newline = "\n"

# vocab parser globas
rank_index = 0
token_index = 1
token_frequency_index = 2

# file i/o
dir_path = "brown/"
dirs = os.listdir(dir_path)
dir_path_normalized = "normalized/"
normalized_dirs = os.listdir(dir_path_normalized)


################################################
################# PRE PROCESSING ###############
################################################

class NormalizeCorpus(object):
    """
    this class helps to:
    - clean the brown corpus
    - saves the corpus to normalized directory, after
        - cleaning tags
        - adding START, STOP, UNK
    - create a sorted_vocabs.txt having the sorted list of tokens
    """

    def __init__(self, percent=100.0):
        self.corpus_percent = percent  # parses upto the given percentage of corpus
        self.token_count = defaultdict(int)  # maintains count of tokens
        self.normalize()  # cleans the corpus and inserts START / STOP
        self.replace_unk()  # replaces lower frequency words with UNK
        self.dump_sorted_count()  # dumps the words counts to sorted_vocabs.txt

    def normalize(self):
        """
        :return:  cleans file from brown corpus and stores it to normalized directory
        """
        file_count = len(dirs)
        file_index = 1
        for file in dirs:
            percent_done = float(file_index) / float(file_count) * 100
            if percent_done > self.corpus_percent:
                break
            # print percent_done
            if self.is_valid_file(file):
                filepath = dir_path + file
                with open(filepath) as f:
                    # read content
                    contents = f.read().replace(newline, '').replace(tab_delim, '').lower()

                    # normalize content
                    normalized_content = self.normalize_content(contents)

                    # save to file
                    ff = open(dir_path_normalized + file, 'w')
                    ff.write(normalized_content)
                    ff.close()
                    # break
            file_index += 1

    @staticmethod
    def is_valid_file(file):
        """
        :param file:
        :return: to check if the file name is valid
        """
        return len(file) == 4

    def normalize_content(self, contents):
        """
        :param contents:
        :return:  replaces tags from each word token
        """
        normalized_sentences = []
        sentences = contents.split("./.")
        # print "sentences count: ", len(sentences)
        for sentence in sentences:
            ns = self.normalize_helper(sentence)
            normalized_sentences.append(ns)

        normalized_content = ''.join(normalized_sentences)
        return normalized_content

    def normalize_helper(self, sentence):
        """
        :param sentence:
        :return: helper for the above method. Creates a string with normalized tokens
        """
        raw_tokens = sentence.split(space_delim)
        tokens = ["START "]
        for rt in raw_tokens:
            t = rt.split(hash_delim)[0]
            if self.is_valid_token(t):
                self.token_count[t] += 1
                tokens.append(t + space_delim)

        tokens.append('STOP')
        tokens.append(newline)
        return ''.join(tokens)

    @staticmethod
    def is_valid_token(k):
        """
        :param k:
        :return: filter for alpha numeric words
        """
        rs = re.search('[a-zA-Z0-9]', k)
        if rs is None:
            return False
        else:
            return True

    def replace_unk(self):
        """
        :return: converts the cleaned up brown corpus further, by inserting UNK symbols
        """
        self.token_count["START"] = 11
        self.token_count["STOP"] = 11
        for file in normalized_dirs:
            filepath = dir_path_normalized + file
            f1 = open(filepath, 'r')
            contents = f1.read()
            f1.close()

            f2 = open(filepath, 'w')
            final_content = self.get_unk_replaced_content(contents)
            f2.write(final_content)
            f2.close()

        # pprint.pprint(self.token_count)
        # print self.token_count['the']
        # print self.token_count['UNK']

    def get_unk_replaced_content(self, contents):
        """
        :param contents:
        :return: helper for the above method
        """
        unk = "UNK"
        sentences = contents.split(newline)
        unked_content = []
        for sentence in sentences:
            tokens = sentence.split(space_delim)
            if len(tokens) < 3:
                continue
            unk_sentence = []
            for token in tokens:
                if self.token_count[token] <= 10:
                    self.token_count[unk] += 1
                    unk_sentence.append(unk + space_delim)
                else:
                    unk_sentence.append(token + space_delim)
            unked_content.append(''.join(unk_sentence) + newline)
        return ''.join(unked_content)

    def dump_sorted_count(self):
        '''
        :return: parses the corpus and creates a map of the word and frequency count
        '''
        tups = self.token_count.items()
        tups.sort(key=lambda x: (-x[1], x[0]))
        # write to file
        f = open('sorted_vocabs.txt', 'w')
        rank = 1
        for word, freq in tups:
            if freq <= 10:
                # word = "UNK"
                continue
            f.write(str(rank) + tab_delim + word + tab_delim + str(freq) + newline)
            rank += 1
        f.close()


class CorpusAnalyser(object):
    """
    this class helps to:
    - create a count map of all tokens
    - create a bigram table for all pairs of takens in the corpus
        - self.bigram_table[apple][the] gives the tatal count of occurreces where 'the' came before 'apple'
    """

    def __init__(self):
        self.bigram_table = {}
        self.count_map = defaultdict(int)
        self.build_count_map()
        self.build_bigram_table()

    def build_count_map(self):
        """
        :return: rebuilds the count map of word and frequency
        """
        for file in normalized_dirs:
            filepath = dir_path_normalized + file
            with open(filepath) as f:
                for line in f:
                    tokens = line.split(space_delim)
                    for token in tokens:
                        self.count_map[token] += 1
        print "\t>> token cound map calculated of size:", len(self.count_map)

    def build_bigram_table(self):
        """
        :return: stores the count for bigram table.
        For example, self.bigram_table[apple][the] gives the tatal count of occurreces where 'the' came before 'apple'
        """
        for file in normalized_dirs:
            filepath = dir_path_normalized + file
            with open(filepath) as f:
                for line in f:
                    tokens = line.split(space_delim)
                    for i in range(1, len(tokens) - 1):
                        prev = tokens[i]
                        curr = tokens[i + 1]
                        if curr in self.bigram_table:
                            d = self.bigram_table[curr]
                            d[prev] += 1
                        else:
                            d = defaultdict(int)
                            d[prev] += 1
                            self.bigram_table[curr] = d
        print "\t>> bigram table calculated of size:", len(self.bigram_table)

# parse sorted_vocab to reconstruct token_frequency map
class VocabParser(object):
    def __init__(self):
        self.last_rank = -1
        self.ranked_token_map = {}
        self.build_vocab_parser()
        # self.last_rank = self.get_last_ranked_token()

    def get_ranked_token(self, rank):
        """
        :param rank:
        :return: returns the token with the given rank
        """
        if rank in self.ranked_token_map:
            return self.ranked_token_map[rank][0]
        else:
            print "token of rank %s not found", rank

    def get_ranked_tuple(self, rank):
        """
        :param rank:
        :return: returns a tuple of (rank, token, frequency)
        """
        if rank in self.ranked_token_map:
            return self.ranked_token_map[rank]
        else:
            print "token of rank %s not found", rank

    def build_vocab_parser(self):
        """
        :return: reconstructs the map after parsing through sorted_vocabs.txt
        """
        filepath = "sorted_vocabs.txt"
        with open(filepath) as f:
            for line in f:
                tokens = line.split(tab_delim)
                rank = int(tokens[rank_index])
                token = tokens[token_index]
                frequency = int(tokens[token_frequency_index])
                self.ranked_token_map[rank] = (token, frequency)
                self.last_rank = rank

    def get_last_ranked_token(self):
        """
        :return: returns the rank of the last rank
        """
        rank = 1
        token = ""
        while True:
            token = self.get_ranked_token(rank + 1)
            if token == "UNK":
                return rank
            rank += 1
        return -1


################################################
################# DATA STRUCTURE ###############
################################################
class ClusterNode(object):
    """
    - this class models a cluster
    - this class object, cluster node is then used by the class defined below, cluster tree.
    - the implementation is similar to a binary tree,
        but each node contains a list of information defined in the constructor below
    """

    def __init__(self, index, key=None):
        self.index = index
        self.key = key
        if key is None:
            self.leaf_values = []
        else:
            self.leaf_values = [key]
        self.left = None
        self.right = None

    def is_leaf(self):
        """
        :return: returns true if its a leaf node
        """
        if self.key is None:
            return False
        else:
            return True

    def get_cluster_leaf_values(self):
        """
        :return: returns the list of all tokens that belong to the cluster
        """
        return self.leaf_values


class ClusterTree(object):
    """
    - this class models the binary cluster tree
    - nodes of the tree are cluster node objects
    - the tree also has helper methods to print binary encodings
    """
    def __init__(self):
        self.cluster_map = {}
        self.cluster_size = 0
        self.encoding_map = {}

    def get_cluster_by_index(self, index):
        """
        :param index:
        :return: returns cluster, for a given index
        """
        return self.cluster_map[index]

    def add_cluster(self, index, cluster):
        """
        :param index:
        :param cluster:
        :return: adds a new cluster with the given index
        """
        if index in self.cluster_map:
            print "cluster index %s already occupied, overwriting", index
        self.cluster_map[index] = cluster
        self.cluster_size = len(self.cluster_map)

    def merge_clusters(self, index_i, index_j):
        """
        :param index_i:
        :param index_j:
        :return: merges two clusters with indices, index_i and index_j
            ** The merge happens at index_i
        """
        # step1: merge cluster i and j, at index i
        new_cluster = ClusterNode(index_i)
        new_cluster.left = self.cluster_map[index_i]
        new_cluster.right = self.cluster_map[index_j]
        new_cluster.leaf_values = self.cluster_map[index_i].leaf_values + self.cluster_map[index_j].leaf_values
        self.cluster_map[index_i] = new_cluster  # update the cluster at index i

        # step2: decrement index of all indices by 1, for all indices after j
        for i in range(index_j + 1, self.cluster_size + 1):
            if i in self.cluster_map:
                cluster_i = self.cluster_map[i]
                self.cluster_map[i - 1] = cluster_i
                # print "updated key, ", i - 1
                # last_index = i - 1

        # step3: last index
        last_index = self.cluster_size
        if last_index in self.cluster_map:
            # print "deleting index: ", last_index
            del self.cluster_map[last_index]

        # step4: update cluster size
        self.cluster_size = len(self.cluster_map)

    def map_all_encodings(self):
        """
        :return: this method traverses the tree pre order, to construct binary encodings
        """
        if self.cluster_size != 1:
            print "encoding not possible right now, final merge the cluster tree"
        else:
            print "starting complete."
            encoding_string = ""
            self.encoding_helper(self.cluster_map[1], encoding_string)
            print "encoding complete."

    def encoding_helper(self, root, encoding_string):
        """
        :param root:
        :param encoding_string:
        :return: recursive helper for the above function
        """
        if root is None:
            return

        # if its a leaf, update the encoding for the key
        if root.is_leaf():
            # self.encoding_map[root.key] = encoding_string
            self.encoding_map[encoding_string] = root.get_cluster_leaf_values()
        else:
            self.encoding_helper(root.left, encoding_string + "0")
            self.encoding_helper(root.right, encoding_string + "1")

    def get_all_leaf_values(self):
        """
        :return: returns the list of all tokens held inside the cluster
        """
        return self.cluster_map[1].leaf_values


################################################
###### BROWN CLUSTERING IMPLEMENTATION #########
################################################
class BrownClustering(object):
    """
    - class having the driver code to run the brown clustering algorithm
    - its initializes all the other class instances and pre computes values for quality optimization
    """
    def __init__(self, k, random_flag=True):
        self.random_flag = random_flag
        self.k = k

        self.ca = CorpusAnalyser()
        self.bigram_table = self.ca.bigram_table
        self.count_map = self.ca.count_map
        self.total_token_count = self.find_total_token_count()

        self.vp = VocabParser()
        self.ct = ClusterTree()

        # algorithm drivers
        self.run_algorithm()
        self.print_encodings()

    def run_algorithm(self):

        # step1: create k top clusters
        self.create_top_k_clusters()
        print "step1: created k clusters"

        # step2: loop from k+1...V
        print "step2: loop through vocab"
        self.loop_through_vocab()

        # step3: merge remaining clusters
        self.merge_remaining()
        print "step3: merge remaining"

    def create_top_k_clusters(self):
        m = self.k + 1
        for rank in range(1, m):
            ranked_token = self.vp.get_ranked_token(rank)
            cn = ClusterNode(rank, ranked_token)
            self.ct.add_cluster(rank, cn)
            # print "added cluster", ranked_token

    def print_encodings(self):
        f = open('encoding_map.txt', 'w')
        self.ct.map_all_encodings()
        import pprint
        # pprint.pprint(self.ct.encoding_map)
        with f as fp:
            fp.write(pprint.pformat(self.ct.encoding_map))
        f.close()
        print "binary map stored to encoding_map.txt"

    def calculate_quality(self, index1, index2):

        if self.random_flag:
            # if random flag is turned on, quality is calculated as a random number
            return random.randint(1, 999999999)
        else:
            Q = 0.0
            end = self.k + 1
            for i in range(1, end):
                for j in range(1, end):

                    # cases for index1, index2 pairs
                    if j == index2:
                        # index 2 does not exist as it has been merged with index 1
                        continue
                    if i == index1:
                        # calculate quality 'assuming' clusters 1 and 2 have been merged
                        return self.calculate_quality_for_new_clusters(Q, index1, index2, j)

                    # other cases
                    # optimization: if results have already been computed,
                    # return quality directly
                    if index1 in self.quality_cache:
                        if index2 in self.quality_cache[index1]:
                            return self.quality_cache[index1][index2]

                    # else compute quality again, and store it in cache
                    c1 = self.ct.get_cluster_by_index(i)
                    c2 = self.ct.get_cluster_by_index(j)

                    pc1 = self.calculate_pc(c1)
                    pc2 = self.calculate_pc(c2)
                    pcc = self.calculate_pcc(c1, c2)

                    if pcc == 0:
                        Q = 0
                    else:
                        Q += pcc * math.log10(pcc / (pc1 * pc2))

            # store result to cache
            self.quality_cache[index1] = {index2: Q}
            return Q

    def calculate_quality_for_new_clusters(self, Q, index1, index2, other_cluster_index):
        n = self.total_token_count
        cluster_i_leaves = self.ct.get_cluster_by_index(index1).get_cluster_leaf_values()
        cluster_j_leaves = self.ct.get_cluster_by_index(index2).get_cluster_leaf_values()
        cluster_i_j_leaves = cluster_i_leaves + cluster_j_leaves  # temporary merge of clusters

        # pc1
        cp1 = 0
        for token in cluster_i_j_leaves:
            cp1 += self.count_map[token]
        cp1 = float(cp1)
        pc1 = cp1 / n

        # pc2
        c2 = self.ct.get_cluster_by_index(other_cluster_index)
        pc2 = self.calculate_pc(c2)

        # pcc
        pcc = 0
        current_token_list = cluster_i_j_leaves
        previous_token_list = c2.get_cluster_leaf_values()
        for t1 in current_token_list:
            for t2 in previous_token_list:
                pcc += self.bigram_table[t1][t2]
        pcc = float(pcc)
        pcc /= n

        # calculate quality
        if pcc == 0:
            return 0
        Q = pcc * math.log10(pcc / (pc1 * pc2))
        return Q

    def loop_through_vocab(self):
        for rank in range(self.k + 1, self.vp.last_rank):
            print "\tcurrent iteration: %s / %s " % (rank, self.vp.last_rank)
            if rank % 500 == 0:
                print "iteration: ", rank, self.vp.last_rank
            # add a cluster of rank k+1
            ranked_token = self.vp.get_ranked_token(rank)
            cn = ClusterNode(rank, ranked_token)
            self.ct.add_cluster(self.k + 1, cn)  # size of cluster = k + 1

            # max_quality = float("-inf")
            max_quality = 0.0
            for i in range(1, self.k + 2):
                self.quality_cache = {}
                for j in range(i + 1, self.k + 2):
                    # print "\t iteration: ", rank, i, j
                    cluster1 = self.ct.get_cluster_by_index(i)
                    cluster2 = self.ct.get_cluster_by_index(j)
                    quality = self.calculate_quality(i, j)

                    if quality > max_quality:  # update indices, max_quality found so far
                        index1, index2 = i, j
                        max_quality = quality

            # merge indices index1, index2
            self.ct.merge_clusters(index1, index2)  # size of cluster back to k

    def merge_remaining(self):
        while self.ct.cluster_size > 1:
            self.ct.merge_clusters(1, 2)

    def calculate_pc(self, c):
        cp = 0
        for token in c.get_cluster_leaf_values():
            cp += self.count_map[token]
        cp = float(cp)
        n = self.total_token_count
        return cp / n

    def calculate_pcc(self, c1, c2):
        pcc = 0
        current_token_list = c1.get_cluster_leaf_values()
        previous_token_list = c2.get_cluster_leaf_values()

        for t1 in current_token_list:
            for t2 in previous_token_list:
                pcc += self.bigram_table[t1][t2]

        pcc = float(pcc)
        n = self.total_token_count
        return pcc / n

    def find_total_token_count(self):
        n = 0
        for word, freq in self.count_map.iteritems():
            n += freq
        return float(n)




# default values
k = 200
random_flag = False

# argument parsing
n = len(sys.argv)
if n == 1:
    pass
if n == 3:
    k = sys.argv[2]
if n == 4:
    k = sys.argv[2]
    random_flag = True

k = int(k)
print "k value: %s, random flag for quality calculation: %s" %(k, random_flag)

print "starting brown clustering algorithm"
time.sleep(1)

# test corpus normalizer
print "normalizing corpus... "
nc = NormalizeCorpus()
print "corpus cleaned, stored in normalized/ directory"

# test brown clustering results
bc = BrownClustering(k, random_flag)
print "clustering complete, binary encoding saved in encoding_map.txt"


"""
# tests for classes

# test brown clustering results
bc = BrownClustering(200)

# test corpus analyzer
ca = CorpusAnalyser()

# test corpus normalizer
nc = NormalizeCorpus()
print "done !"

# test vocab parser
vb = VocabParser()
print len(vb.ranked_token_map)
print vb.ranked_token_map[1]

# test cluster
c1 = ClusterNode(1, "the")
c2 = ClusterNode(2, "cat")
c3 = ClusterNode(3, "dog")
# print c1.is_leaf()
# print c2.is_leaf()

# test cluster tree
ct = ClusterTree()
ct.add_cluster(1, c1)
ct.add_cluster(2, c2)
ct.add_cluster(3, c3)
ct.merge_clusters(1, 2)
ct.merge_clusters(1, 2)
ct.map_all_encodings()
# print "cluster_size:", ct.cluster_size
# print "cluster_map: ", ct.cluster_map
print ct.encoding_map
"""
