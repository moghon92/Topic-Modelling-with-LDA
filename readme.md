# Topic Modeling

##  Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is a generative statistical model that explains a set of observations through unobserved groups, and each group explains why some parts of the data are similar. The LDA is an example of a topic model. In this, observations (e.g., words) are collected into documents, and each word's presence is attributable to one of the document's topics. Each document will contain a small number of topics.

## In this project we 
- Lowercase, tokenize and de-accent sentences to produce the tokens using simple_preprocess function from gensim.utils
- Remove stopwords from tokenized words.
- Create dicitionary and term document frequency for the input data using Dicitonary class of gensim.corpora.
- Build LDA Model using LdaMulticore class of gensim.models.
