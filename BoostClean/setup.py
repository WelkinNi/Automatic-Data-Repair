from setuptools import setup, find_packages

setup(
  name = 'activedetect',
  packages = find_packages(), # this must be the same as the name above
  version = '0.1.4-2',
  description = 'A Library For Error Detection For Predictive Analytics',
  author = 'Sanjay Krishnan and Eugene Wu',
  author_email = 'sanjay@eecs.berkeley.edu',
  url = 'https://github.com/sjyk/activedetect/', # use the URL to the github repo
  download_url = 'https://github.com/sjyk/activedetect/tarball/0.1.4-2', 
  keywords = ['error', 'detection', 'cleaning'], # arbitrary keywords
  classifiers = [],
  install_requires=[
          'numpy',
          'sklearn',
          'gensim',
          'usaddress',
          'scipy'
      ]
)
