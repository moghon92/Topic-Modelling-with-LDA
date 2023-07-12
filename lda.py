import gensim


class LDA:
    def __init__(self):
        """
        Initialize LDA Class
        """
        # self.lda_model = gensim.models.ldamodel.LdaModel()

    def tokenize_words(self, inputs):
        """
        Lowercase, tokenize and de-accent sentences to produce the tokens using simple_preprocess function from gensim.utils.

        Args:
            inputs: Input data.

        Returns:
            output: Tokenized list of sentences.
        """

        return [gensim.utils.simple_preprocess(doc, deacc=True) for doc in inputs]

    def remove_stopwords(self, inputs, stop_words):
        """
        Remove stopwords from tokenized words.

        Args:
            inputs: Input data.
            stop_words: List of stop_words

        Returns:
            output: Tokenized list of sentences.
        """
        out = []
        for sent in inputs:
            tok_sent = []
            for word in sent:
                if word not in stop_words:
                    tok_sent.append(word)
            out.append(tok_sent)
        return out

    def create_dictionary(self, inputs):
        """
        Create dicitionary and term document frequency for the input data using Dicitonary class of gensim.corpora.

        Args:
            inputs: Input data.

        Returns:
            id2word: Index to word map.
            corpus: Term document frequency for each word.
        """

        id2word = gensim.corpora.Dictionary(inputs)
        corpus = [id2word.doc2bow(sent) for sent in inputs]

        return id2word, corpus

    def build_LDAModel(self, id2word, corpus, num_topics=10):
        """
        Build LDA Model using LdaMulticore class of gensim.models.

        Args:
            id2word: Index to word map.
            corpus: Term document frequency for each word.
            num_topics: Number of topics for modeling

        Returns:
            lda_model: LdaMulticore instance.
        """

        return gensim.models.ldamulticore.LdaMulticore(corpus, id2word=id2word, num_topics=num_topics)


