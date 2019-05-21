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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

#from custom_scoring import mo_confusion_matrix, mo_weighted_cm_scorer,mo_weighted_average_precision, make_scorers




def load_data_single(y_set):

    X = pd.read_pickle(r'../data/Xinput.pkl')
    Y = pd.read_pickle(r'../data/Y%s.pkl' % y_set)

    return(X,Y)


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
    #     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
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

    pipe_LogisticReg = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression())),
    ])
    par_LogisticReg = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__C': [1,0.8,0.5,.2],
    }

    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """

    y_pred = model.predict(X_test)
    classification_report = pd.DataFrame(
        classification_report(Y_test.values, y_pred, target_names=category_names,
                              output_dict=True)).T

    print(classification_report)


def save_model(model, model_filepath):
    pass

def udacity_main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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



def main():

    col='food'

    X, Y  = load_data_single(col)
    print('database loaded')
    print('X:', X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    vect = CountVectorizer(tokenizer=tokenize, min_df=2, max_df=1.0, max_features=2000 , ngram_range=(1,4))
    vect.fit(X_train)
    X_train1= vect.transform(X_train)
    X_test1= vect.transform(X_test)

    tfidf = TfidfTransformer()
    tfidf.fit(X_train1)
    X_train2= tfidf.transform(X_train1)
    X_test2= tfidf.transform(X_test1)


    pipe_LogisticReg = Pipeline([
        ('clf', LogisticRegression(random_state=0, solver='liblinear',penalty='l1',   ={0:1,1:50})),
    ])
    par_LogisticReg = {

        'clf__estimator__C': [1,0.8,0.7,0.5,0.4,0.3,.2,0.1,0.01],
        'clf__estimator__class_weight': [{1:w} for w in [1,2,5,10,20,30,40,50,100]  ]
    }

    scorer='W1'


    filename='%s_%s_%s' % (col,scorer)
    print(filename)

    model = pipe_LogisticReg
    parameters =  par_LogisticReg
    cv_folds=3

    scorerAP = make_scorer(met.average_precision_score, greater_is_better = True)

    #cv = GridSearchCV(model, param_grid=parameters, verbose=2, cv=cv_folds)
    cv = GridSearchCV(model, scoring=scorerAP, param_grid=parameters, verbose=2, cv=cv_folds, refit='AP')
    cv.fit(X_train2, y_train2)

    #print(cv)

    print()
    print('training finished')
    print()

    pickle.dump(cv, open(filename + '_cvAll.pkl', 'wb'))


    pickle.dump(cv.best_estimator_, open(filename + '_model.pkl', 'wb'))
    pickle.dump(cv.best_params_, open(filename + '_params.pkl', 'wb'))
    pickle.dump(y_test, open(filename + '_ytest.pkl', 'wb'))

    y_pred = cv.best_estimator_.predict(X_test2)
    pickle.dump(y_pred, open(filename + '_ypred.pkl', 'wb'))



if __name__ == '__main__':
    main()
