# CSCI 404 Liu Winter 2018 Homework I\V
# Haley Beavers & Emmanuel Harley


# FORMATTERS ###################################
sep = "========================================="
yep = "-----------------------------------------"
################################################

# IMPORTS ######################################
import nltk
from nltk.probability import FreqDist
from nltk.tokenize    import word_tokenize

import sys
import os
import numpy as np
################################################


# FUNCTION DEFINITIONS #########################
# gathers initial, transitions, and emissions counts
# for HMM from training set
def count(filename):
    # two-tiered, single-tiered
    initial     = {}
    emissions   = {}
    transitions = {}
    tag_dict    = {}
    wrds = []
    tags = []

    data_path = filename
    file      = open(data_path, "r")
    line      = file.readlines()

    prev = '###'
    for tokens in line:
        tokens  = tokens.strip("\r\n")
        wrd,tag = tokens.split('/')
        wrds.append(wrd)
        tags.append(tag)

        # INITIAL
        if prev == '###':
            # incr word count assoc w/ tag
            if tag in initial:
                initial[tag] += 1
            else:
                initial[tag] = 1

        # TRANSITIONS
        coor = (prev,tag)
        if coor in transitions:
            transitions[coor] += 1
        else:
            transitions[coor]  = 1

        # EMISSIONS
        # adding tags to emissions
        if tag not in emissions:
            emissions[tag] = {}

        # incr word count assoc w/ tag
        if wrd in emissions[tag]:
            emissions[tag][wrd] += 1
        else:
            emissions[tag][wrd]  = 1

        if wrd in tag_dict:
            if tag not in tag_dict[wrd]:
                tag_dict[wrd] += (tag,)
        else:
            tag_dict[wrd] = ()

        prev = tag

    wrds = dict(FreqDist(wrds))
    tags = dict(FreqDist(tags))
    return (initial,emissions,transitions, wrds, tags, tag_dict)

################################################
# adding OOV tags for novel words encountered in test set
def add_oov(tag_dict, em, tr, words, tags):
    for i in range(len(words)):
        if words[i] not in em[tags[i]]:
            if 'OOV' not in em[tags[i]]:
                em[tags[i]]['OOV'] = 1
                words[i] = 'OOV'
            else:
                em[tags[i]]['OOV'] += 1
                words[i] = 'OOV'

        if words[i] not in tag_dict:
            tag_dict[words[i]] = (tags[i],)
        elif tags[i] not in tag_dict[words[i]]:
            tag_dict[words[i]] += (tags[i],)

    novel_tags = []
    for i in range(1, len(tags)):

        temp = (tags[i-1], tags[i])
        if temp not in tr:
            tr[temp] = 1
            novel_tags.append(temp)
        elif temp in novel_tags:
            tr[temp] += 1

        temp = (tags[i], tags[i-1])
        if temp not in tr:
            tr[temp] = 1
            novel_tags.append(temp)
        elif temp in novel_tags:
            tr[temp] += 1

    return tag_dict, em, tr, words

################################################
# implementation of Viterbi algorithm
def viterbi(wrd_seq, counts):
    init, em, tr, wrds, tags, tag = counts
    n_words = len(wrd_seq)
    n_tags  = len(tags)
    tag_keys = tags.keys()

    bp_tags = ["" for x in range(n_words)]
    backpointer = [{} for x in range(n_words)]
    u_seqs = [{} for x in range(n_words)]

    u_seqs[0]['###'] = 1.
    backpointer[0]['###'] = 1.

    for i in range(1,n_words):
        for ti in tag_dict[wrd_seq[i]]:
            for ti_1 in tag_dict[wrd_seq[i-1]]:
                # calculate viterbi probabilities from the previous word
                prob = np.log(p_tt(em, tr, ti, ti_1)) + np.log(p_tw(em, ti, wrd_seq[i]))

                # find the maximum value of the viterbi probabilities
                u = np.log(u_seqs[i-1][ti_1]) + prob
                if ti not in u_seqs[i]:
                    u_seqs[i][ti] = u
                    backpointer[i][ti] = ti_1
                elif u > u_seqs[i][ti]:
                    u_seqs[i][ti] = u

    # calculate the last viterbi probability from the previous word
    bp_tags[n_words - 1] = '###'

    for i in range(n_words - 1, 0, -1):
        bp_tags[i - 1] = backpointer[i][bp_tags[i]]

    return bp_tags

################################################
# gets words and tags in given file and puts in list
def test(path):
    words = []
    tagss = []

    file  = open(path, "r")
    lines = file.readlines()

    for line in lines:
        wrd_tag = line.strip("\r\n").split('/')

        words.append(wrd_tag[0])
        tagss.append(wrd_tag[1])

    return words, tagss

################################################
# number of singleton words for any given tag
def sing_tw(emissions):
    sing = 0
    tags = emissions.keys()
    for tag in tags:
        counts = emissions[tag].values()
        for count in counts:
            if count == 1:
                sing += 1

    return sing

################################################
# number of singleton transitions for any given tag
def sing_tt(transitions):
    sing = 0
    counts = transitions.values()
    for count in counts:
        if count == 1:
            sing += 1

    return sing

################################################
# return (count_word + 1) / (n + V)
def backoff_tw(w_i, emissions):
    w = 0 # count of word
    v = 0 # number of types
    n = 0 # number of tokens

    tags = emissions.keys()
    for tag in tags:
        v += 1
        words = emissions[tag].keys()
        for word in words:
            count = emissions[tag][word]
            if w_i == word:
                w += count
            n += count

    return float(w + 1) / float(n + v)

################################################
# return count_tag / (number_word_tokens)
def backoff_tt(tag_2, emissions):
    t = 0
    n = 0
    flag = False
    tags = emissions.keys()

    for tag in tags:
        if tag_2 == tag:
            flag = True
        else:
            flag = False
        counts = emissions[tag].values()
        for count in counts:
            n += count
            if flag == True:
                t += count

    return float(t) / float(n)

################################################
# counts number of times tag appears
def count_tag(emissions, ti):
    t = 0
    tags = emissions.keys()

    for tag in tags:
        counts = emissions[tag].values()
        if ti == tag:
            flag = True
        else:
            flag = False
        for count in counts:
            if flag == True:
                t += count

    return t

################################################
# calcuating backoff probability for emissions
def p_tw(emissions, tag, word):
    l  = 1 + sing_tw(emissions)
    p  = backoff_tw(tag, emissions)
    tw = emissions[tag][word]
    t  = count_tag(emissions, tag)

    return float(tw + l) * float(p) / float(t + l)

################################################
# calcuating backoff probability for transitions
def p_tt(emissions, transitions, tag_1, tag_2):
    if transitions.has_key((tag_1,tag_2)) == False:
        transitions[(tag_1,tag_2)] = 1

    l  = 1 + sing_tt(emissions)
    p  = backoff_tt(tag_2, emissions)
    tt = transitions[(tag_1,tag_2)]
    t  = count_tag(emissions, tag_2)

    return float(tt + l) * float(p) / float(t + l)

################################################
# finds most likely tag sequence given word sequence
def decode(wrd_seq, tags, tags_dict):
    wrd_seqs = make_seq(wrd_seq)
    tag_seqs = make_seq(tags)
    n_sep_seq = len(wrd_seqs)

    for i in range(n_sep_seq):
        path = viterbi(wrd_seqs[i], counts)

        correct       = 0
        known_correct = 0
        known_total   = 0
        novel_correct = 0
        novel_total   = 0
        n_tags = len(path)
        for j in range(n_tags):
            #print(path[j], tag_seqs[i][j], wrd_seqs[i][j])
            if 'OOV' == wrd_seqs[i][j]:
                if path[j] == tag_seqs[i][j]:
                    novel_correct += 1
                    correct       += 1
                novel_total   += 1
            else:
                if tag_seqs[i][j] != '###':
                    if path[j] == tag_seqs[i][j]:
                        correct += 1
                        known_correct += 1
                    known_total   += 1

        # calculating tagging accuracy
        acc = float(correct)/float(n_tags - 2)
        known_acc = float(known_correct)/float(known_total)
        if novel_total != 0:
            novel_acc = float(novel_correct)/float(novel_total)
        else:
            novel_acc = 0.
        print('Tagging accuracy (Viterbi decoding): {0:.2f}% (known: {1:.2f}% novel: {2:.2f}%)'.format(acc, known_acc, novel_acc))

################################################
# seperate sequences in given
def make_seq(wrd_seq):
    ls  = []
    seq = []
    j   = 0

    seq.append("###")
    for i in range(1,len(wrd_seq)):
        if wrd_seq[i] == "###":
            seq.append(wrd_seq[i])
            ls.append(seq)
            seq = ["###"]
            i += 1
        else:
            seq.append(wrd_seq[i])

    return ls

################################################
#
def raw(filename):
    data_path = filename
    file      = open(data_path, "r")
    lines     = file.readlines()

    v = []
    for line in lines:
        v.append(line.strip("\r\n"))

    return v

################################################


# EXERCISES ####################################

# 1. Data Sets
path_ic = "./data/ic/"
path_en = "./data/en/"

words,tags = test(path_en + 'entest.txt')

# getting and storing counts
counts = count(path_en + 'entrain.txt')
initial, emissions, transitions, wrds, tags_dict, tag_dict = counts

rawv = raw("./data/en/enraw.txt")
v = list(set().union(wrds,rawv))
v_size = len(v) + 1

tag_dict, emissions, transitions, words = add_oov(tag_dict, emissions, transitions, words, tags)

decode(words, tags, tags_dict)
"""
# 1. Data Sets
path_ic = "./data/ic/"
path_en = "./data/en/"

# user inputs filepath to test and train
path  = sys.argv[1]
test  = sys.argv[2]
train = sys.argv[3]

filename = path + test
words,tags = test(path_en + 'entest.txt')

# calculating and upacking counts
filename = path + train
counts = count(path_en + 'entrain.txt')
initial, emissions, transitions, wrds, tags_dict, tag_dict = counts

# getting raw vocabulary set if specified
raw = ""
if raw != "":
    rawv = raw("./data/en/enraw.txt")
    v = list(set().union(wrds,rawv))
    v_size = len(v) + 1

# decoding and calculating tagging accuracy
tag_dict, emissions, transitions, words = add_oov(tag_dict, emissions, transitions, words, tags)
decode(words, tags, tags_dict)
"""
################################################
