#!/usr/bin/env python

# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for printing the problem at the end
# Thanks for fixing: lary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
# This program takes a set of documents and clusters them into a set of topics.
# The topics are then printed out in a list. This is a program that you can use to see the topics of a set of documents.
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the value error
# Thanks for fixing the code. I'm very grateful it works now
# I fixed this. Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing the code. I was worried I didn't have any of the right files
# Thanks for fixing: lary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
# Thanks for fixing: lary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)

from __future__ import print_function

import argparse
import json
import sys

from time import time

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

from sklearn.decomposition import TruncatedSVD

from sklearn.utils import extmath
from sklearn.decomposition import NMF, LatentDirichletAllocation
from random import randint

categories = ['alt.atheism',
'comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'misc.forsale',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'soc.religion.christian',
'talk.politics.guns',
'talk.politics.mideast',
'talk.politics.misc',
'talk.religion.misc']


def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
            dest="n_components", type="int",
            help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
            action="store_false", dest="minibatch", default=True,
            help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
            action="store_false", dest="use_idf", default=True,
            help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
            action="store_true", default=False,
            help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
            help="Maximum number of features (dimensions)"
                 " to extract from text.")
    op.add_option("--verbose",
            action="store_true", dest="verbose", default=False,
            help="Print progress reports inside k-means algorithm.")

    print(__doc__)
    op.print_help()

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)


    ###############################################################################
    # Load some categories from the training set
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)
    
    
    dataset = load_files(container_path='aws', categories=categories, load_content=True, shuffle=True, encoding='utf-8')


    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()

    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                    stop_words='english', non_negative=True,
                    norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                    stop_words='english',
                    non_negative=False, norm='l2',
                    binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])
    print(vectorizer.get_feature_names()[randint(0,len(vectorizer.get_feature_names()))])

if __name__ == "__main__":
    sys.exit(main())
