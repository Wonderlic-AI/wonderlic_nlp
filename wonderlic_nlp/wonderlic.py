# Built-in imports
import re
import statistics as stats
import string
import logging
import os

# Third-party imports
import enchant
import numpy as np
import pandas as pd
import spacy
import textstat
from empath import Empath

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from scipy.spatial import distance
from textblob import TextBlob, Word

from sklearn.feature_extraction.text import TfidfVectorizer

# Homemade imports
from wonderlic_nlp.psycholinguistics import *

class WonderlicNLP:
    def __init__(self, 
        enchant_lang = 'en_US', 
        spacy_model = 'en_core_web_sm'):

        # Global constants
        self.ENCHANT_LANG = enchant_lang
        self.SPACY_MODEL = spacy_model

        # Reusable objects
        self.__lexicon = Empath()
        self.__mrcdb = MRCDatabase()
        self.__subdb = SubtlexDatabase()
        self.__tfidf = TfidfVectorizer()
        self.__tfidf_fit = False

        try:
            self.__nlp = spacy.load(self.SPACY_MODEL)

        except Exception as e:
            logging.warning("Spacy model '{}' not installed. Installing now.".format(self.SPACY_MODEL))
            os.system('python -m spacy download {}'.format(self.SPACY_MODEL))
            self.__nlp = spacy.load(self.SPACY_MODEL)

        self.__enchant_dict = enchant.Dict(self.ENCHANT_LANG)


    def __merge(self, dictionaries):
        first = dictionaries[0]
        
        for next_ in dictionaries[1:]:
            first.update(next_)

        return first

    def analyze(self, text):
        return self.__merge([{
                'general':
                    self.__merge([self.get_num_words(text),
                    self.get_num_sentences(text),
                    self.conduct_sentiment_anaylsis(text)]),
                'mrc': self.get_mrc_features(text),
                'topic_signals': self.get_topic_signals(text),
                'punct': self.get_punct_features(text)
                },
                self.get_reading_level(text),
                self.get_pos_bag(text)])

    # `get_subtlex_from_word`: get subtlex word form frequency and contextual diversity from word
    # - Inputs: word (str)
    # - Returns dict
    def get_subtlex_from_word(self, word):
        return {'Word Form Frequency': self.__subdb.get_subtl_wf(word),
                'Contextual Diversity': self.__subdb.get_subtl_cd(word)}


    # `tfidf_fit_transform`: Learn vocabulary and idf, return document-term matrix.
    # - Inputs: list of documents (str)
    # - Returns document-term matrix (numpy aray)
    def tfidf_fit_transform(self, list_of_text):
        self.__tfidf_fit = True
        return self.__tfidf.fit_transform(list_of_text)


    # `tfidf_transform`: Transform documents to document-term matrix.
    # - Inputs: list of documents (str)
    # - Returns document-term matrix (numpy aray)
    def tfidf_transform(self, list_of_text):
        if self.__tfidf_fit:
            return self.__tfidf.transform(list_of_text)
        else:
            logging.error('The TF-IDF vectorizer is not fitted')


    # `correct_spelling`: makes a best attempt to correct spelling in a given string of text, and returns count of misspelled words
    # - Inputs: text (str)
    # - Returns str, int

    def correct_spelling(self, text):
        doc = self.strip_special_whitespace(text)
        doc = self.strip_punctuation(doc)
        words = doc.split()
        quick_fixes = {word: self.__reduce_lengthening(word) for word in words
                       if len(self.__reduce_lengthening(word)) != len(word)}
        for word, fix in quick_fixes.items():
            text = text.replace(word, fix)

        doc = TextBlob(text)

        return str(doc.correct())

    # `get_num_words`: count the number of words in a text document
    # - Inputs: document (str)
    # - Returns dict

    def get_num_words(self, document):
        return {'num_words': len(self.strip_punctuation(document).split())}


    # `get_num_sentences`: count the number of sentences in a text document
    # - Inputs: document (str)
    # - Returns dict

    def get_num_sentences(self, document):
        return {'num_sentences': len(sent_tokenize(document))}

    # `get_avg_sentence_len`: calculate the average number of words per sentence in a text document
    # - Inputs: document (str)
    # - Returns dict

    def get_avg_sentence_len(self, document):
        num_words = self.get_num_words(document)['num_words']
        num_sentences = self.get_num_sentences(document)['num_sentences']
        return {'avg_sent_len': num_words / num_sentences}

    # `get_topic_signals`: uses a topic engine (e.g. Empath) to extract a count of topic occurences in text
    # - Inputs: document (str), engine (str)
    # - Returns dict

    def get_topic_signals(self, document, engine='empath'):
        if engine == 'empath':
            topic_counts = self.__lexicon.analyze(document, normalize=False)
        else:
            raise ValueError('engine must be \'empath\'')

        return {engine + '_' + topic: value for topic, value in topic_counts.items()}

    # `get_mrc_features`: collects and aggregates psycholinguistic features from the MRC database for words in the provided document
    # - Inputs: document (str)
    # - Returns dict

    def get_mrc_features(self, document):
        accessors = self.__get_mrc_accessors(self.__mrcdb)
        feature_tracker = {feature: [] for feature in accessors}

        doc = self.strip_special_whitespace(document)
        doc = self.strip_punctuation(doc)
        doc = self.__nlp(doc)
        for token in doc:
            if token.pos_ in ('SPACE', 'SYM', 'EOL'):
                continue
            word = token.text
            pos_tuple = get_mrc_pos(token.pos_)
            for feature in accessors:
                for pos in pos_tuple:
                    feature_value = accessors[feature](word, pos)
                    if feature_value:
                        break
                if feature_value:
                    # if type(feature_value) is str:
                    #     print(feature)
                    #     print(feature_value)
                    feature_tracker[feature].append(feature_value)

        mrc_analysis = {}
        for feature, tracker in feature_tracker.items():
            if len(tracker) > 0:
                mrc_analysis[feature] = stats.mean(tracker)
            else:
                mrc_analysis[feature] = 0 # Temporary for AOA None due to sparsity error.

        return mrc_analysis

    # `__get_mrc_accessors`: retrieves a mapping of MRC features to their getter methods
    # - Inputs: mrc_database (MRCDatabase obj)
    # - Returns dict

    def __get_mrc_accessors(self, mrc_database):
        return {'Nlet': mrc_database.get_num_letters,
                'Nphon': mrc_database.get_num_phonemes,
                'Nsyl': mrc_database.get_num_syllables,
                'K-F-freq': mrc_database.get_kf_freq,
                'K-F-ncats': mrc_database.get_kf_num_categories,
                'K-F-nsamp': mrc_database.get_kf_num_samples,
                'T-L-freq': mrc_database.get_tl_freq,
                'Brown-freq': mrc_database.get_brown_freq,
                'Fam': mrc_database.get_familiarity,
                'Conc': mrc_database.get_concreteness,
                'Imag': mrc_database.get_imageability,
                'Meanc': mrc_database.get_colorado_meaningfulness,
                'Meanp': mrc_database.get_paivio_meaningfulness,
                'AOA': mrc_database.get_age_of_acquisition}


    # ## Pre-processing & helper functions

    # `strip_special_whitespace`: removes all '\t' and '\n' escape characters from the provided text
    # - Inputs: text (str)
    # - Returns str

    def strip_special_whitespace(self, text):
        return text.translate(str.maketrans('', '', '\n\t'))


    # `strip_punctuation`: removes all punctuation from the provided text
    # - Inputs: text (str)
    # - Returns str

    def strip_punctuation(self, text):
        return text.translate(str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'))


    # `remove_stopwords`: strips a string of all word tokens matching a set of stop words
    # - Inputs: text (str)
    # - Returns str

    def remove_stopwords(self, text):
        stops = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_tokens = [word for word in word_tokens if not word in stops]

        return ' '.join(filtered_tokens)


    # `lemmatize`: converts words in a text string to their meaningful base forms
    # - Inputs: text (str)
    # - Returns str

    def lemmatize(self, text):
        doc = self.__nlp(text)
        lemma_tokens = [token.lemma_ for token in doc]

        return ' '.join(lemma_tokens)


    # `__average_over_sentences`: converts sentence-level, dict-returning feature extractions to document-level
    # - Inputs: document (str), sentence_function (function), remove_punctuation (bool)
    # - Returns dict

    def __average_over_sentences(self, document, sentence_function, remove_punctuation=False):
        sentences = sent_tokenize(document)
        if remove_punctuation:
            sentences = [strip_punctuation(sentence) for sentence in sentences]

        aggregator = None
        sent_num = 0
        for sentence in sentences:
            sent_num += 1
            sentence_features = sentence_function(sentence)
            if not aggregator:
                aggregator = {key: [value] if value else []
                              for key, value in sentence_features.items()}
            else:
                for key in sentence_features:
                    if sentence_features[key]:
                        aggregator[key] = aggregator.get(key, []) + [sentence_features[key]]

        return {feature: stats.mean(tracker) if len(tracker) > 0 else None
                for feature, tracker in aggregator.items()}


    # `__reduce_lengthening`: forces any group of a single repeated letter down to no more than two letters (e.g. "bubbble" -> "bubble")
    # - Inputs: text (str)
    # - Returns str

    def __reduce_lengthening(self, text):
        return re.compile(r"(.)\1{2,}").sub(r"\1\1", text)


    # `conduct_sentiment_analysis`: computes the average sentence polarity metric across a text document
    # - Inputs: document (str)
    # - Returns dict

    def conduct_sentiment_anaylsis(self, document):
        blob = TextBlob(document)
        polarities = [sentence.sentiment.polarity for sentence in blob.sentences]

        return {'average_polarity': stats.mean(polarities)}

    # `get_pos_bag`: create a "bag of tags" for the parts-of-speech tagged in the document
    # - Inputs: document (str)
    # - Returns dict

    def get_pos_bag(self, document):
        bag_of_pos = {}
        doc = self.__nlp(document)
        for token in doc:
            bag_of_pos[token.pos_] = bag_of_pos.get(token.pos_, 0) + 1

        return {'bag_of_pos': bag_of_pos}


    # `get_bag_of_words`: create a key-value mapped bag of words
    # - Inputs: document (str)
    # - Returns dict

    def get_bag_of_words(self, document):
        doc = self.strip_special_whitespace(document)
        doc = self.strip_punctuation(doc)
        bag_of_words = {}
        for word in word_tokenize(doc):
            bag_of_words[word.lower()] = bag_of_words.get(word, 0) + 1

        return {'bag_of_words': bag_of_words}


    # `get_punct_features`: count punctuation marks used throughout the document
    # - Inputs: document (str)
    # - Returns dict

    def get_punct_features(self, document):
        puncts = [char for char in document if char in string.punctuation]
        punct_features = {'Period': puncts.count('.'),
                          'Comma': puncts.count(','),
                          'Colon': puncts.count(':'),
                          'Semic': puncts.count(';'),
                          'Qmark': puncts.count('?'),
                          'Exclam': puncts.count('!'),
                          'Dash': puncts.count('-'),
                          'Quote': puncts.count('"'),
                          'Apostro': puncts.count('\''),
                          'Parenth': puncts.count('(') + puncts.count(')')}
        punct_features['Otherp'] = len(puncts) - sum(punct_features.values())
        punct_features['Allpct'] = len(puncts)

        return {'punct': punct_features}


    # `get_reading_level`: determine the approximate reading level of the document based on a consensus of the Flesch Reading Ease formula, the Flesch-Kincaid Grade Level, the Fog Scale, the SMOG Index, the Automated Readability Index, the Coleman-Liau Index, the Linsear Write Formula, and the Dale-Chall Readability Score
    # - Inputs: document (str)
    # - Returns dict

    def get_reading_level(self, document):
        return {'reading_level': textstat.text_standard(document, float_output=True)}


    # `do_spell_check`: runs a spell checker on the text and returns a count and ratio of misspelled words
    # - Inputs: document (str)
    # - Returns dict

    def do_spell_check(self, document):
        doc = self.strip_special_whitespace(document)
        doc = self.strip_punctuation(doc)
        words = doc.split()
    
        num_misspelled = 0
        misspelled_words = []
        for word in words:
            if not self.__enchant_dict.check(word):
                misspelled_words.append(word)
                num_misspelled += 1

        return {'count_misspelled': num_misspelled, 
                'ratio_misspelled': num_misspelled / len(words),
                'misspelled_words': misspelled_words}
