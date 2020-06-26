from distutils.core import setup
setup(
  name = 'wonderlic_nlp',
  packages = ['wonderlic_nlp'],
  version = '0.1.1',
  license='gpl-3.0',
  description = 'General-purpose NLP toolkit for accessing popular psycholinguistics databases.',
  author = 'Ross Piper, Abdallah Aboelela',
  author_email = 'ross.piper@wonderlic.com, abdallah.aboelela@wonderlic.com',
  url = 'https://github.com/Wonderlic-AI/wonderlic_nlp',
  download_url = 'https://github.com/Wonderlic-AI/wonderlic_nlp/archive/v_0_1_1.tar.gz',
  keywords = ['NLP', 'Python', 'Natural', 'Language', 'Processing', 'English', 'Python', 'tfidf', 'MRC', 'SUBTLEX', 'SUBTLEXus', 'Psycholinguistics'],
  install_requires=[            # I get to this in a second
          'pandas',
          'scipy',
          'textblob',
          'spacy',
          'nltk',
          'empath',
          'numpy',
          'pyenchant',
          'sklearn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'Programming Language :: Python :: 3.7',
  ],
)