# NLP
Projects completed in Natural Language Processing course, all done as pair programming assignments with Robert Harley (xmannyh). 


### Edit-Dist
Implemented Levenshtein Distance Algorithm, given two strings displays n different edit distance configurations between them using operations delete, replace, and subsitute. A matrix is created that can be travesed from the bottom left corner to upper right, each path represents a different edit configuration between the two strings, or different way to operate on the strings to get from one to the other.

### N-Gram
Wrote N-Gram Language Model Algorithm, given a sample text word sequence probabilities are stored and used to generate text using simple Baysian statistics in uni, bi, tri, or n-grams modeling.

### Naïve-Bayes
Implemented Naïve Bayes Algorithm to classify e-mails as spam or non-spam. Given a corpus of spam, and non-spam e-mails use Bayesian statistics to classify never before seen e-mails as spam or non-spam. A probabilstic classifier that can be used for an arbitrary number of classes.

### Viterbi-Decode
Implemented Hidden Markov Model (HMM) and decoded probability of sequence of events using the Viterbi Algorithm. Gathered counts for emission, transition, and initial probabilities from training set of sequences, then applied these learned probabilities onto never before seen sequences to predict how likely they are to occur.
