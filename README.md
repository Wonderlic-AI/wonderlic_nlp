# Wonderlic NLP

**Wonderlic NLP** is a general-purpose natural language processing (NLP) toolkit in Python that contains easy-to-use interfaces for accessing popular psycholinguistics databases. Currently MRC and SUBTLEXus are supported.

This package is supported and maintained by the [Wonderlic AI team](https://wonderlic.ai/). To learn more about our latest AI and innovation projects, check out https://wonderlic.ai/. For more information on Wonderlic and WonScore, see https://wonderlic.com/

## Installation

To install Wonderlic NLP download it from [`PyPI`](https://pypi.org/project/#/).

```python
pip install wonderlic_nlp
```

## Usage

### Analyze
```python
from wonderlic_nlp import WonderlicNLP

wnlp = WonderlicNLP()
wnlp.analyze(text) # Returns a combined dictionary of all of the below functions

wnlp.get_mrc_features(text) # get MRC features, see https://websites.psychology.uwa.edu.au/school/MRCDatabase/uwa_mrc.htm
wnlp.conduct_sentiment_anaylsis(text) # get average sentence polarity
wnlp.get_topic_signals(text) # get dictionary of Empath topic signals
wnlp.get_avg_sentence_len(text) # get average sentence length
wnlp.get_punct_features(text) # get punctuation counts
wnlp.get_pos_bag(text) # get parts of speech counts
wnlp.get_num_sentences(text) # get number of sentences
wnlp.get_num_words(text) # get number of words
```

### Functions on text
```python
wnlp.do_spell_check(text) # get number of misspelled words, proportion of words misspelled, and a list of misspelled words
wnlp.correct_spelling(text) # get text with spelling corrected
wnlp.get_bag_of_words(text) # get bag of words
wnlp.get_tfidf_vector(text) # get tfidf vector
wnlp.lemmatize(text) # converts words in a text string to their meaningful base forms
wnlp.remove_stopwords(text) # strips a string of all word tokens matching a set of stop words
wnlp.strip_punctuation(text) # removes all punctuation from the provided text
wnlp.strip_special_whitespace(text) # removes all '\t' and '\n' escape characters from the provided text

wnlp.tfidf_fit_transform(list_of_documents) # Train TF-IDF model, Returns numpy sparse matrix
wnlp.tfidf_transform(list_of_documents) # Transform using TF-IDF model, numpy sparse matrix
```

### Functions on invidiual words
```python
wnlp.get_subtlex_from_word(word) # get SUBTLEXus word form frequency and contextual diversity
wnlp.get_mrc_features(word) # get MRC features for individual word
```

## Other
Wonderlic NLP is built on Spacy and Enchant. It defaults to "en_US" for Enchant language, and "en_core_web_sm" for the Spacy model. To use a different model or language, simply pass those into the WonderlicNLP constructor.

```python
modif_enchant_wnlp = WonderlicNLP(enchant_lang = 'other_enchant_lang')
modif_spacy_wnlp = WonderlicNLP(spacy_model = 'other_spacy_model')
```