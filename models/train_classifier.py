import sys
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer

rom sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

import sklearn.metrics as met
from sklearn.utils.fixes import signature
import pickle

def load_data(database_filepath):
    """
    loads the data from the SQL database
    :param database_filepath: SQL database path
    :return: X,Y,category_names
    """

    # database_filepath='sqlite:///data//InsertDatabaseName.db'
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names=list(df.columns)

    return(X,Y,category_names)


def stem(word):
    ps = PorterStemmer()
    stemmer_exceptions = ['sos']
    if word not in stemmer_exceptions:
        return ps.stem(word)
    else:
        return word


def tokenize(text):
    """
    tokenizes the messagages into words (lowercase, remove punctuation, split, remove stopwords and lemmatize with WordNet)
    Inputs:
    - text: string

    """

    stop_extra = ['http', 'afghanistan', 'afghan', 'africa', 'african', 'asia', 'asian', 'australia', 'australian',
                  'balochistan', 'bamako', 'bangladesh',
                  'british', 'brooklyn', 'canada', 'canadian', 'ethiopia', 'ethiopian', 'french', 'haitian', 'haitien',
                  'india', 'indian', 'individu',
                  'indonesia', 'islamabad', 'jakarta', 'jersey', 'kabul', 'kachipul', 'kandahar', 'karachi', 'kashmir',
                  'kenya', 'kenyan', 'liberia',
                  'korea', 'niger', 'nigeria', 'pakistan', 'pakistani', 'somali', 'somalia', 'sudan', 'taiwan',
                  'tajikistan', 'tanzania',
                  'thailand', 'uganda', 'vietnam', 'zimbabw', 'www', 'xinhua', 'yangtz', 'klecin', 'bernadett', 'croix',
                  'mercredi',
                  'wednesday', 'rue', 'saint', 'santo', 'thank']

    text = text.replace("S.O.S.", "SOS")
    text = text.lower()

    # remove numbers, name of places,and common stopwords
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_extra]
    words = [w for w in words if w not in stopwords.words("english")]

    words = [w for w in words if len(w) > 2]

    words = [stem(w) for w in words]

    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    return (lemmed)

def nlp_pipeline(X_train,X_test,min_df=2, max_df=1.0, max_features=10000, ngram_range=(1, 4)):
    # splitting the nlp from the logistic regression pipeline results in a significant speedup

    vect = CountVectorizer(tokenizer=tokenize, min_df=min_df, max_df=max_df, max_features=max_features,
                           ngram_range=ngram_range)
    vect.fit(X_train)
    X_train1 = vect.transform(X_train)
    X_test1 = vect.transform(X_test)

    tfidf = TfidfTransformer()
    tfidf.fit(X_train1)
    X_train2 = tfidf.transform(X_train1)
    X_test2 = tfidf.transform(X_test1)

    return(X_train2,X_test2, vect, tfidf )

def build_full_model():

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            LogisticRegression(random_state=0, solver='liblinear', penalty='l1', max_iter=200, )
        )),
    ])
    parameters= {
        'vect__ngram_range': ((1, 1), (1, 4)),
        'vect__max_df': (0.9, 1.0),
        'vect__min_df': (2,3),
        'vect__max_features': (2000, 5000,10000, 20000, None),

        'clf__estimator__C': [1,   0.5,0.3, 0.25 ,.2,0.15, 0.1 ],
        'clf__estimator__class_weight': [{1:w} for w in [1,5,10,20,30,50]  ]
    }
    cv_folds = 5

    scorerAP = make_scorer(met.average_precision_score, greater_is_better=True)
    cv = GridSearchCV(model, scoring=scorerAP, param_grid=parameters, verbose=1, cv=cv_folds, refit='AP')
    return(cv.best_estimator_)

def build_model():
    '''
    fits a logistic regression model and returns the best model

    :return:
    '''
    model = Pipeline([
            ('clf', LogisticRegression(random_state=0, solver='liblinear',penalty='l1',max_iter=200, )),
        ])
    parameters = {
        'clf__C': [1,   0.5,0.3, 0.25 ,.2,0.15, 0.1 ],
        'clf__class_weight': [{1:w} for w in [1,5,10,20,30,50]  ]
    }

    cv_folds = 5

    scorerAP = make_scorer(met.average_precision_score, greater_is_better=True)
    cv = GridSearchCV(model, scoring=scorerAP, param_grid=parameters, verbose=1, cv=cv_folds, refit='AP')
    return(cv.best_estimator_)

def evaluate_model(model, X_test, Y_test, category_names):



    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X_train, X_test, vect, tfidf = nlp_pipeline(X_train, X_test,max_features=2000)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier_perf_single.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()