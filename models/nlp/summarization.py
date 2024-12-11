import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        """
        Initialize the text summarizer.
        """
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = string.punctuation

    def score_sentences(self, text):
        """
        Score sentences based on the frequency of significant words.
        :param text: The input text to summarize.
        :return: Dictionary with sentence indices as keys and scores as values.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)

        # Count word frequencies, ignoring stop words and punctuation
        for word in words:
            if word not in self.stop_words and word not in self.punctuation:
                word_freq[word] += 1

        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]

        return sentence_scores

    def summarize(self, text, n=2):
        """
        Generate a summary by selecting the top-n scored sentences.
        :param text: The input text to summarize.
        :param n: Number of sentences to include in the summary.
        :return: Summary as a string.
        """
        sentence_scores = self.score_sentences(text)
        top_sentences = sorted(sentence_scores.keys(), key=lambda k: sentence_scores[k], reverse=True)[:n]
        sentences = sent_tokenize(text)

        # Reconstruct summary with top sentences in original order
        summary = [sentences[i] for i in sorted(top_sentences)]
        return " ".join(summary)
