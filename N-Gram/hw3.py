# CSCI 404 Liu Winter 2018 Homework III
# Haley Beavers & Emmanuel Harley


# FORMATTERS ###################################
sep = "========================================="
yep = "-----------------------------------------"
################################################


# IMPORTS ######################################
import nltk
from nltk.probability import FreqDist
from nltk.tokenize    import sent_tokenize
from nltk.tokenize    import word_tokenize
from nltk.tokenize    import WordPunctTokenizer
from nltk.tokenize    import TweetTokenizer
from nltk.util 		  import ngrams

import numpy as np
import random as rdm
import collections
################################################

#controls how long the sentences can be.
threshold = 15
#controls the number of sentences to generate
n_sentences_generate = 10

# FUNCTION DEFINITIONS #########################
# adding all lines in file to string list
def file_text(filename):
	file      = open(filename, "r")
	lines     = file.readlines()
	sentences = []
	for line in lines:
		sentences.append(line.strip("\r\n"))
	return sentences

################################################
# adds <s>, </s> at beginning and end of sentences
# replaces all tokens with count of one with <UNK>
def add_sym(corpus):
	words = []
	for sentence in corpus:
		words+=  ['<s>'] + word_tokenize(sentence) + ['</s>']

	unigrams = dict(FreqDist(words))
	n_words = len(words)
	n_unknown = 0
	for i in range(n_words):
		if unigrams[words[i]] == 1:
			del unigrams[words[i]]
			words[i] = '<UNK>'
			n_unknown+= 1

	unigrams['<UNK>'] = n_unknown

	return (unigrams, words)

################################################
# computes complexity for a unigram, bigram, or trigram given for a k value of 0, 1, 2 respectively
def perplexity(n_grams, words, k):
	unigrams = n_grams[0]
	bigrams = n_grams[1]
	trigrams = n_grams[2]

	n_tokens = len(words)
	vocab = unigrams.keys()
	vocab_size = len(vocab)
	probs = []
	if k == 0:
		probs+= [float(unigrams[word] + 1)/float(n_tokens + vocab_size) for word in words]
	elif k == 1:
		for i in range(1, n_tokens):
			probs.append((float(bigrams[(words[i-1], words[i])] + 1))/(float(unigrams[words[i-1]] + vocab_size)))
	elif k == 2:
		for i in range(2, n_tokens):
			bigram = (words[i-2], words[i-1])
			trigram = (words[i-2], words[i-1], words[i])
			probs.append((float(trigrams[trigram] + 1))/(float(bigrams[bigram] + vocab_size)))

	pp = np.exp(-1.0 * np.sum(np.log(probs)) / float(n_tokens))
	return pp

################################################
# picks random next unigram
def next_uni(tokens):
	index = np.random.randint(0,len(tokens))
	return tokens[index]

################################################
# constructs sentence using unigram probabilities
def generate_uni(unigrams, uni_dict):
	common_words = dict(FreqDist(unigrams).most_common(30))

################################################
# computes log probabilities of generated sentence
def log_prob(sentence, words):
	# bigram
	# np.log2(bigrams[(sentence[0],sentence[1]] / unigrams[sentence[0]])
	vocab_size = len(unigrams)
	n_tokens = np.sum(unigrams.values())
	n_swords = len(sentence)
	probs = []

	ngrams = [dict(FreqDist(nltk.ngrams(words, i))) for i in range(1, n_swords)]
	for i in range(n_swords - 1, 0, -1):
		try:
			num = float(ngrams[i-1][tuple(sentence[:i])] + 1)
		except:
			num = 1.0
		try:
			denom = float(ngrams[i-2][tuple(sentence[:i-1])] + vocab_size)
		except:
			denom = float(vocab_size)

		probs.append(num/denom)
	#calculate the last unigram term for the probability (should always be <s>...)
	probs.append(float(unigrams[sentence[0]] + 1) / float(n_tokens + vocab_size))


	prob = np.sum(np.log(probs))
	return prob

################################################
# constructs sentence using unigrams
def generate_uni(unigrams, words):
	common_words = dict(FreqDist(words).most_common(30))
	# run into problem if <s> not in 30 most common
	# deleting key that DNE
	try:
		del common_words['<s>']
	except:
		pass

	common_words = common_words.keys()

	sentence = []
	for i in range(n_sentences_generate):
		sentence.append('<s>')
		while sentence[len(sentence)-1] != '</s>' and len(sentence) < threshold:
			next_word = next_uni(common_words)
			while next_word == '</s>' and len(sentence) == 1:
				next_word = next_uni(common_words)
			sentence.append(next_uni(common_words))

		print(yep)
		print(str(i+1) + ". " + str(sentence))

		n = len(sentence)
		prob = log_prob(sentence, uni_dict, unigrams)
		print("LOG PROBABILITY: " + str(prob))
		if sentence[len(sentence)-1] != '</s>':
			sentence.append('</s>')

		prob_sentence = log_prob(sentence, words)
		print(yep)
		print('{0}) {1}\nlog-space prob: {2:.2f}'.format(i+1, str(sentence), prob_sentence))
		sentence = []

################################################
# finds most probable bigram given previous token
def next_bi(prev_token, unigrams, bigrams):
	candidates = []
	for unigram in unigrams:
		bigram = (prev_token, unigram)
		if bigram in bigrams:
			candidates.append(bigram[1])

	rand_idx = np.random.randint(0,len(candidates))
	return candidates[rand_idx]

################################################
# constructs sentence using bigram probabilities
def generate_bi(unigrams, bigrams):
	sentence = []

	for i in range(n_sentences_generate):
		sentence  = ['<s>']
		last_unigram = sentence[len(sentence) - 1]

		# add next probable bigram until length reached, or eos token
		while last_unigram != '</s>' and len(sentence) < threshold:
			next = next_bi(last_unigram, unigrams, bigrams)
			sentence.append(next)
			last_unigram = sentence[len(sentence)-1]

		# if length attribute reached first, need to append
		if last_unigram != '</s>':
			sentence.append('</s>')
		print(yep)
		print(str(i+1) + ". " + str(sentence))

		n = len(sentence)
		prob = log_prob(sentence, unigrams)
		print("LOG PROBABILITY: " + str(prob))

		prob_sentence = log_prob(sentence, words)
		print(yep)
		print('{0}) {1}\nlog-space prob: {2:.2f}'.format(i+1, str(sentence), prob_sentence))
		sentence = []

################################################
# finds most probable bigram given previous token
def first_tri(trigrams):
	candidates = []

	for trigram in trigrams:
		if trigram[0] == '<s>':
			candidates.append(trigram[1])

	rand_idx = np.random.randint(0,len(candidates))
	return candidates[rand_idx]

################################################
# finds most probable bigram given previous token
def next_tri(prev_tokens, trigrams):
	candidates = []
	prev_1 = prev_tokens[0]
	prev_2 = prev_tokens[1]

	for trigram in trigrams:
		if trigram[0] == prev_1 and trigram[1] == prev_2:
			candidates.append(trigram[2])

	rand_idx = np.random.randint(0,len(candidates))
	return candidates[rand_idx]

################################################
# dont look ples :(
def tri_sentence(trigrams):
	try:
		sentence = ['<s>']
		last_bigram = ('<s>', first_tri(bigrams))

		while last_bigram[1] != '</s>' and len(sentence) <  threshold:
			next_word = next_tri(last_bigram, trigrams)
			sentence.append(next_word)
			last_bigram = (sentence[len(sentence)-2], sentence[len(sentence)-1])
		# if length attribute reached first, need to append
		if last_bigram[1] != '</s>':
			sentence.append('</s>')
	except:
		# no judgement
		sentence = tri_sentence(trigrams)

	return sentence

################################################
# constructs sentence using trigram probabilities
def generate_tri(unigrams, bigrams, trigrams):
	for i in range(3):
		sentence = tri_sentence(trigrams)
		print(yep)
		print(str(i+1) + ". " + str(sentence))

		n = len(sentence)
		prob = log_prob(sentence, unigrams)
		print("LOG PROBABILITY: " + str(prob))

################################################
# computes log probabilities of generated sentence
def log_prob(sentence, unigrams, words):
	# bigram
	# np.log2(bigrams[(sentence[0],sentence[1]] / unigrams[sentence[0]])
	ng = ngrams(words, 2)
	counted = collections.Counter(ng)
	prev_grams = dict(counted)

	# special case, unigram log probability
	init_prob = np.log2(unigrams[sentence[0]] / len(unigrams))
	if (len(sentence) == 3):
		return init_prob
	else:
		return rec_prob(sentence, init_prob, unigrams, 2)

################################################
# recursively calculates log probabilities
def rec_prob(sentence, prob, prev_grams, n):
	 # base case
	 if n == len(sentence)-1:
		 return np.log2(ngram[sentence[:n-1]] / prev_grams[sentence[:n]])
	 else:
		model = ngrams(sentence, n)
		prob * rec_prob(sentence, prob, model, n+1)

	for i in range(n_sentences_generate):
		sentence = tri_sentence(trigrams)
		prob_sentence = log_prob(sentence, words)
		print(yep)
		print('{0}) {1}\nlog-space prob: {2:.2f}'.format(i+1, str(sentence), prob_sentence))
		sentence = []

################################################


# READING IN FILES #############################
# training corpus
filename = 'data/train.txt'
train = file_text(filename)

# test corpus
filename = 'data/test.txt'
test = file_text(filename)

################################################


# EXERCISES ####################################
# 1
# Extracting vocabulary
vocabulary = []
# words contains each word in sentence
words = []
num_sent = len(train)
for sentence in train:
	words += word_tokenize(sentence)

# adding <s>, </s>, and <UNK>
counts_words = add_sym(sad)
unigrams     = counts_words[0]
words        = counts_words[1]
n_words = len(words)

# unique tokens in training data
vocabulary = unigrams.keys()
print("1.a Vocabulary Size: " + str(len(vocabulary)))

# Generate Unigram Model
print("1.b Unigram Model Generated")
print('Unigram count: ' + str(len(unigrams)))


# Generate Bigram Model
print("1.c Bigram Model Generated")
#bigrams = dict(FreqDist(nltk.bigrams(words)))

# Generate Trigram Model
print("1.d Trigram Model Generated")
#trigrams = dict(FreqDist(nltk.trigrams(words)))

bigrams = dict(FreqDist(nltk.bigrams(words)))
print('Bigram count: ' + str(len(bigrams)))

# Generate Trigram Model
print("1.d Trigram Model Generated")
trigrams = dict(FreqDist(nltk.trigrams(words)))
print('Trigram count: ' + str(len(trigrams)))
print(sep)

# 2
# Generate Unigram Sentences
print("2.a Generating 10 Unigram Sentences: ")
generate_uni(unigrams, words)
print(yep)

# Generate Bigram Sentences
print("2.b Generating 10 Bigram Sentences: ")
generate_bi(unigrams.keys(), bigrams)
print(yep)

# Generate Trigram Sentences
print("2.c Generating 10 Trigram Sentence: ")
generate_tri(unigrams.keys(), bigrams, trigrams)


print(sep)

generate_tri(unigrams.keys(), bigrams, trigrams)
print(sep)

# 3
# Compute Perplexity
print("3 Perplexity of Language Models: ")
pp1 = perplexity([unigrams, bigrams, trigrams], words, 0)
pp2 = perplexity([unigrams, bigrams, trigrams], words, 1)
pp3 = perplexity([unigrams, bigrams, trigrams], words, 2)
print('Unigram PP: {0:.2f}'.format(pp1))
print('Bigram PP: {0:.2f}'.format(pp2))
print('Trigram PP: {0:.2f}'.format(pp3))
if pp1 > pp2:
	if pp2 > pp3:
		print('we started from the bottom now we\'re here')
################################################
