# CSCI 404 Liu Winter 2018 Homework IV
# Haley Beavers & Emmanuel Harley


# FORMATTERS ###################################
sep = "========================================="
yep = "-----------------------------------------"
################################################

vocab_size = 2500

# IMPORTS ######################################
import nltk
from nltk.probability import FreqDist
from nltk.tokenize    import word_tokenize

import os
import numpy as np
################################################


# FUNCTION DEFINITIONS #########################
# Return the tokens from the emails in the given folder
def getTrainingTokens(folder, email_limit):
    lines_in_email = []
    c_mail = 0
    for email in os.listdir(folder):
        if c_mail >= email_limit:
            break
        file      = open(folder+email, "r")
        lines     = file.readlines()
        for line in lines:
            lines_in_email.append(line.strip("\r\n"))

        c_mail+= 1

    bag_of_words = []
    for email in lines_in_email:
        bag_of_words += word_tokenize(email)

    return bag_of_words

################################################
# Return the tokens from the emails in the given folder
# every element in the list is a list of words an email
def getTestTokens(folder):
    all_bags_of_words = []

    for email in os.listdir(folder):
        lines_in_email = []
        file      = open(folder+email, "r")
        lines     = file.readlines()
        for line in lines:
            lines_in_email.append(line.strip("\r\n"))

        bag_of_words = []
        for email in lines_in_email:
            bag_of_words += word_tokenize(email)

        all_bags_of_words.append(bag_of_words)

    return all_bags_of_words

################################################
# calculating probaility that word in document appears in each class
def class_prob(token, c, d_spam, d_nspam, train_totals):
    #If the key is not in the most probable list ignore it.
    if token not in d_spam or token not in d_nspam:
        return 1.0/float(vocab_size)

    if c == 0:
        return float(d_nspam[token] + 1) / float(train_totals[0] + vocab_size)
    else:
        return float(d_spam[token]  + 1) / float(train_totals[1] + vocab_size)

################################################
# returns the summed log probabilites of emails under classifier
def log_prob(probs):
    return np.sum(np.log(probs[i]))

################################################
# displays false-true-positive-negative table
def table(fp, tp, fn, tn):
    pipe = "|"
    undr = "_" * (len("false") + 3 + len("negative")*2)

    print((" " * len("false")) + pipe + "negative" + pipe + "positive|")
    print(undr)

    fn_1 = " " + fn + " "
    tn_1 = " " + fp + " "
    fp_1 = " " + tn + " "
    tp_1 = " " + tp + " "

    pad_1 = (" " * (len("negative") - len(fn_1)))
    pad_2 = (" " * (len("positive") - len(fp_1)))
    pad_3 = (" " * (len("positive") - len(tn_1)))
    pad_4 = (" " * (len("positive") - len(tp_1)))


    print("false" + pipe + fn_1 + pad_1 + pipe + fp_1 + pad_2 + pipe)
    print(undr)
    print("true " + pipe + tn_1 + pad_3 + pipe + tp_1 + pad_4 + pipe)
    print(undr)

# classifier
def classify(training_doc_count):
    # EXERCISES ####################################
    # 1 Gather Data

    # TRAIN ########################################
    path_train_ns = "./data/nonspam-train/"
    path_train_s = "./data/spam-train/"
    
    # Collecting Non Spam Training Data
    train_words_ns = getTrainingTokens(path_train_ns, training_doc_count)
    # Collecting Spam Training Data
    train_words_s = getTrainingTokens(path_train_s, training_doc_count)

    uni_train_nspam = dict(FreqDist(train_words_ns).most_common(vocab_size))
    uni_train_spam = dict(FreqDist(train_words_s).most_common(vocab_size))

    # TEST #########################################
    path_test_ns = "./data/nonspam-test/"
    path_test_s = "./data/spam-test/"

    # Collecting Non Spam Test Data
    # contains list of words in each non-spam email
    test_words_ns = getTestTokens(path_test_ns)
    # Collecting Spam Test Data
    # contains list of words in each spam email
    test_words_s = getTestTokens(path_test_s)



    # putting each document's dict into s and ns list
    test_words = [test_words_ns, test_words_s]
    uni_test   = [[], []]
    for category in range(len(test_words)):
        for email in test_words[category]:
            uni_test[category].append(dict(FreqDist(email)).keys())

    # PROBS ########################################
    # calculating probaility that word in document appears in each class
    false_pos = 0
    true_pos  = 0
    false_neg = 0
    true_neg  = 0

    train_totals = [sum(uni_train_nspam.values()), sum(uni_train_spam.values())]

    # contains tokens in each email for each class
    categories = [test_words_ns, test_words_s]

    # for training BOW, testing BOW
    for category in range(len(categories)):
        # for each label
        for email in categories[category]:
            probs = [[], []]
            for class_idx in range(2):
                # for each document
                    for word in email:
                        word_prob = class_prob(word, class_idx, uni_train_spam, uni_train_nspam, train_totals)
                        probs[class_idx].append(word_prob)

            # calculating log probability of words in document being...
            # non-spam
            probs[0] = np.sum(np.log(probs[0]))
            # spam
            probs[1] = np.sum(np.log(probs[1]))

            # largest probability at that index denotes classification
            true_or_false = np.argmax(probs)
            pos_or_neg = category

            # index 1 indicates spam
            if pos_or_neg == 1:
                if true_or_false == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            # index 0 indicates non-spam
            else:
                if true_or_false == 0:
                    true_neg += 1
                else:
                    false_neg += 1

    # RESULTS ######################################
    print(yep)
    # Displaying Contingency Table
    table(str(false_pos), str(true_pos), str(false_neg), str(true_neg))
    print(yep)
    # Calculating Precision
    precision = true_pos / (true_pos + false_pos)
    print("Precision: %" + str(precision*100))
    print(yep)
    # Calculating Recall
    recall = true_pos / (true_pos + false_neg)
    print("Recall: %" + str(recall*100))
    print(yep)
    # Calculating F
    F = (2 * precision * recall) / (precision + recall)
    print("F: %" + str(F*100))
    print(yep)

################################################


classify(960)
extra_cred_vals = [400, 100, 50]
for extra_cred in extra_cred_vals:
    classify(extra_cred)
