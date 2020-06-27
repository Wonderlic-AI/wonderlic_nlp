import pickle
import os


# General-purpose depickling function
# Inputs: filename (str)
# Returns various types
def _depickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    

# Convert string element of keys in MRC dictionary to all-caps
# Inputs: this_dict (dict)
# Returns a dict
def _make_keys_allcaps(this_dict):
    assert len(this_dict) > 0
    test_key = list(this_dict.keys())[0]
    
    if type(test_key) is tuple:
        return {(key[0].upper(), key[1]): value for key, value in this_dict.items()}
    else:
        return {key.upper():value for key, value in this_dict.items()}


# Convert SpaCy POS tags to best-match MRC POS tags
# Inputs: spacy_pos (str)
# Returns a tuple
def get_mrc_pos(spacy_pos):
    pos_mapping = {'PRON': ('U',),
                   'VERB': ('V', 'P'),
                   'DET': ('J',),
                   'NOUN': ('N',),
                   'ADP': ('R',),
                   'ADJ': ('J',),
                   'CCONJ': ('C',),
                   'AUX': ('V',),
                   'ADV': ('A',),
                   'PART': ('O',),
                   'NUM': ('J',),
                   'CONJ': ('C',),
                   'INTJ': ('I',),
                   'PROPN': ('N',),
                   'SCONJ': ('C',),
                   'X': ('O',)}
    
    return pos_mapping.get(spacy_pos, 'O')


# Generalized superclass for psycholinguistic databases
class PLDatabase:
    
    # Loads the pickled data structure
    # Inputs: pickle_filename (str)
    def __init__(self, pickle_filename):
        self._dict = _make_keys_allcaps(_depickle(pickle_filename))
    
    # Indicates whether a given word is included in the database
    # Inputs: word (str)
    # Returns a bool
    def _is_word_in_db(self, word):
        return word.upper() in [key.upper() for key in self._dict]

    # Generalized property retrieval ("getter") method
    # Input: word (str), feature (str)
    # Returns various types, including None if data is unavaiable
    def _get_property_from_word(self, word, feature):
        if not self._is_word_in_db(word):
            return None
        return self._dict[word.upper()][feature]
    

# Subclass to interact (read-only) with the MRC Psycholinguistic Database
# Data provided by the University of Western Australia
# Constructing an instance will load a copy of the database into memory
class MRCDatabase(PLDatabase):
    

    # Constructor method
    # Calls the superclass constructor
    def __init__(self):
        pickle_filename = __file__[:-3] + '/mrc.pickle'
        super().__init__(pickle_filename)
        
    # Indicates whether a given word is included in the MRC database
    # Inputs: word (str), pos (str)
    # Returns a bool
    def _is_word_in_db(self, word, pos):
        return (word.upper(), pos) in [(key[0].upper(), key[1]) for key in self._dict]
        
    # MRC property retrieval ("getter") method
    # Input: word (str), pos (str), feature (str)
    # Returns various types, including None if data is unavaiable
    def _get_property_from_word(self, word, pos, feature):
        match = self._dict.get((word.upper(), pos))
        
        return match[feature] if match else None
        
    # Expand an MRC part-of-speech tag to its full written form
    # Inputs: pos_indicator (str)
    # Returns a string or None
    @staticmethod
    def _expand_partofspeech(pos_indicator):
        pos_dict = {'N': 'noun', 'J': 'adjective', 'V': 'verb', 'A': 'adverb',
                    'R': 'preposition', 'C': 'conjunction', 'U': 'pronoun',
                    'I': 'interjection', 'P': 'past participle', 'O': 'other'}

        return pos_dict.get(pos_indicator)
    
    # Indicates if a given word is a derivational variant of another in the DB
    # Inputs: word (str), pos (str)
    # Returns a bool or None
    def is_derivational_variant(self, word, pos):
        tq2 = self._get_property_from_word(word, pos, 'TYPE')

        return tq2 == 'Q' if self._is_word_in_db(word, pos) else None
    
    # Indicates if a given word is capitalized in one or more usages
    # Inputs: word (str), pos (str)
    # Returns a bool or None
    def is_capitalized(self, word, pos):
        cap = self._get_property_from_word(word, pos, 'CAP')

        return cap == 'C' if self._is_word_in_db(word, pos) else None

    # Retrieve the age of acqisition score of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_age_of_acquisition(self, word, pos):
        return self._get_property_from_word(word, pos, 'AOA')

    # Retrieve the Brown verbal frequency of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_brown_freq(self, word, pos):
        return self._get_property_from_word(word, pos, 'BFRQ')
    
    # Retrieve the concreteness score of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_concreteness(self, word, pos):
        return self._get_property_from_word(word, pos, 'CNC')
    
    # Retrieve the Kucera-Francis number of categories of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_kf_num_categories(self, word, pos):
        return self._get_property_from_word(word, pos, 'KFCAT')
    
    # Retrieve the Kucera-Francis number of samples of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_kf_num_samples(self, word, pos):
        return self._get_property_from_word(word, pos, 'KFSMP')
    
    # Retrieve the Kucera-Francis written frequency of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_kf_freq(self, word, pos):
        return self._get_property_from_word(word, pos, 'KFFRQ')

    # Retrieve the familiarity rating of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_familiarity(self, word, pos):
        return self._get_property_from_word(word, pos, 'FAM')
    
    # Retrieve the common part-of-speech of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_common_partofspeech(self, word, pos):
        cpos = self._get_property_from_word(word, pos, 'CPOS')
        
        return MRCDatabase._expand_partofspeech(cpos)
    
    # Retrieve the imageability rating of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_imageability(self, word, pos):
        return self._get_property_from_word(word, pos, 'IMG')

    # Retrieve the pronunciation variability of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_pronunciation_var(self, word, pos):
        varp = self._get_property_from_word(word, pos, 'VARP')
        varp_dict = {'O': 'different stress patterns', 
                     'B': 'different phonology'}
        
        return varp_dict.get(varp)
    
    # Retrieve the meaningfulness score (using specified norms) of a given word
    # Inputs: word (str), pos (str), norms (str)
    # Returns an int or None
    def get_meaningfulness(self, word, pos, norms):
        available_norms = ['colorado', 'paivio']
        norms_map = {available_norms[0]: 'CMEAN', available_norms[1]: 'PMEAN'}
        
        if not norms in available_norms:
            raise ValueError('Invalid value for \'norms\'')
        
        return self._get_property_from_word(word, pos, norms_map[norms])
    
    # Retrieve the meaningfulness score (using Colorado norms) of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_colorado_meaningfulness(self, word, pos):
        return self.get_meaningfulness(word, pos, 'colorado')
        
    # Retrieve the meaningfulness score (using Paivio norms) of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_paivio_meaningfulness(self, word, pos):
        return self.get_meaningfulness(word, pos, 'paivio')

    # Retrieve the comprehensive syntactic category of a given word
    # Inputs: word (str), pos (str)
    # Returns a str or None
    def get_comprehensive_syntactic_category(self, word, pos):
        csyn = self._get_property_from_word(word, pos, 'CSYN')

        return MRCDatabase._expand_partofspeech(csyn)

    # Retrieve the number of phonemes of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_num_phonemes(self, word, pos):
        return self._get_property_from_word(word, pos, 'NPHN')

    # Retrieve the morphemic status of a given word
    # Inputs: word (str), pos (str)
    # Returns a str or None
    def get_morphemic_status(self, word, pos):
        mrph = self._get_property_from_word(word, pos, 'MRPH')
        mrph_dict = {'S': 'suffix', 'P': 'prefix', 'H': 'hyphenated',
                     'A': 'abbreviation', 'T': 'multi-word phrasal unit'}

        return mrph_dict.get(mrph)
    
    # Retrieve the contextual status of a given word
    # Inputs: word (str), pos (str)
    # Returns a str or None
    def get_contextual_status(self, word, pos):
        mrph = self._get_property_from_word(word, pos, 'STAT')
        mrph_dict = {'$': 'specialized', 'A': 'archaic', 'C': 'capital', 
                     'D': 'dialect', 'E': 'nonsense', 'F': 'foreign/alien', 
                     'H': 'rhetorical', 'N': 'erroneous', 'O': 'obsolete', 
                     'P': 'poetical', 'Q': 'colloquial', 'R': 'rare', 
                     'S': 'standard', 'W': 'nonce word'}

        return mrph_dict.get(mrph)

    # Retrieve the number of syllables of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_num_syllables(self, word, pos):
        return self._get_property_from_word(word, pos, 'NSYL')
    
    # Retrieve the number of letters of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_num_letters(self, word, pos):
        return self._get_property_from_word(word, pos, 'NLET')
    
    # Retrieve the Thorndike-Lorge written frequency count of a given word
    # Inputs: word (str), pos (str)
    # Returns an int or None
    def get_tl_freq(self, word, pos):
        return self._get_property_from_word(word, pos, 'T-LFRQ')
    
    # Retrieve the plural type of a given word
    # Inputs: word (str), pos (str)
    # Returns a str or None
    def get_plural_type(self, word, pos):
        plur = self._get_property_from_word(word, pos, 'PLUR')
        plur_dict = {'Z': 'plural', 'B': 'both', 'Y': 'singular', 
                     'P': 'plural but acts as singular', 'N': 'no plural'}

        return plur_dict.get(plur)


# Subclass to interact (read-only) with the SUBTLEXus Database
# Data provided by Ghent University
# Constructing an instance will load a copy of the database into memory
class SubtlexDatabase(PLDatabase):
    

    # Constructor method
    # Calls the superclass constructor
    def __init__(self):
        pickle_filename = pickle_filename = __file__[:-3] + '/subtl.pickle'
        super().__init__(pickle_filename) 
    
    # Generalized SUBTL score getter method
    # Inputs: word (str), std_feature (str), log_feature (str), log_scale (bool)
    # Returns a float or None
    def _get_subtl_score(self, word, std_feature, log_feature, log_scale):
        feature_key = std_feature if not log_scale else log_feature
        score = self._get_property_from_word(word, feature_key)
        
        return score if not score else float(score)
        
    # Retrieve the SUBTLEX word form frequency of a given word
    # This metric is the frequency of the word per million words
    # Inputs: word (str), log_scale (bool)
    # Returns a float or None
    def get_subtl_wf(self, word, log_scale=False):
        return self._get_subtl_score(word, 'SUBTLWF', 'Lg10WF', log_scale)
    
    # Retrieve the SUBTLEX contextual diversity of a given word
    # This metric is the percentage of indexed films in which the word occurs
    # Inputs: word (str), log_scale (bool)
    # Returns a float or None
    def get_subtl_cd(self, word, log_scale=False):
        return self._get_subtl_score(word, 'SUBTLCD', 'Lg10CD', log_scale)
