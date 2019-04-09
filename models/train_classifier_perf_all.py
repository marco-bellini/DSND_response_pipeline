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

from custom_scoring import mo_confusion_matrix, mo_weighted_cm_scorer,mo_weighted_average_precision, make_scorers


def load_data(database_filepath):
    """

    :param database_filepath:
    :return:
    """

    # database_filepath='sqlite:///data//InsertDatabaseName.db'


    engine = create_engine(database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names=list(df.columns)

    return(X,Y,category_names)


def tokenize(text):
    """
    tokenizes the messagages into words (lowercase, remove punctuation, split, remove stopwords and lemmatize with WordNet)
    Inputs:
    - text: string

    """

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    return (lemmed)

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

def create_pipelines():
    """

    :return:
    """

    pipe_DecisionTree = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier())),
    ])
    par_DecisionTree = {
        'clf__estimator__min_samples_split': [0.1, .9],
    }

    pipe_RandomForest = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    par_RandomForest = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__n_estimators': [20,50, 100],
    }

    pipe_RandomForest = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    par_RandomForest = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__n_estimators': [20,50, 100],
    }

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

    pipe_MultinomialNB = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB())),
    ])
    par_MultinomialNB = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__alpha': [1,0.5,.0],
    }

    pipe_GradBoost= Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier())),
    ])
    par_GradBoost = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__learning_rate': [0.1,0.05,0.2,0.5],
        'clf__estimator__n_estimators': [100,200,500],
        'clf__estimator__max_depth': [3,5,7],
    }

    if 0:
        pipelines={'RF':[pipe_RandomForest,par_RandomForest] , 'DT':[pipe_DecisionTree,par_DecisionTree],
               'LR':[pipe_LogisticReg,par_LogisticReg],'MN':[pipe_MultinomialNB,par_MultinomialNB],
               'GB':[pipe_GradBoost,par_GradBoost]
               }

    pipelines={#'LR':[pipe_LogisticReg,par_LogisticReg],
               'MN':[pipe_MultinomialNB,par_MultinomialNB],
               #'GB':[pipe_GradBoost,par_GradBoost]

               }



    return(pipelines)


def main():

    database_filepath = 'sqlite:///'+'..//data//InsertDatabaseName.db'
    X, Y, category_names = load_data(database_filepath)
    print('database loaded')
    print('X:', X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    pipelines=create_pipelines()

    weights={'tp':2,'tn':.001,'fn':1,'fp':.1}
    class_weights=y_test.iloc[0,:]*0.0+1
    class_weights['child_alone']=0 # no support data
    class_weights['death']=20
    class_weights['floods']=20
    class_weights['fire']=20
    class_weights['storm']=20
    class_weights['earthquake']=20
    class_weights['search_and_rescue'] = 10
    class_weights['medical_help'] = 10
    class_weights['shelter'] = 10

    scorers=make_scorers(weights,class_weights)

    c=2
    # for scorer in scorers.keys():
    for classifier in pipelines.keys():
        print(classifier)
        filename='%d_%s' % (c,classifier)
        print(filename)

        model = pipelines[classifier][0]
        parameters = pipelines[classifier][1]
        cv_folds=5


        #cv = GridSearchCV(model, param_grid=parameters, verbose=2, cv=cv_folds)
        cv = GridSearchCV(model, scoring=scorers, param_grid=parameters, verbose=2, cv=cv_folds, refit='AP')
        cv.fit(X_train, y_train)

        #print(cv)

        print()
        print('training finished')
        print()

        pickle.dump(cv, open(filename + '_cvAll.pkl', 'wb'))


        pickle.dump(cv.best_estimator_, open(filename + '_model.pkl', 'wb'))
        pickle.dump(cv.best_params_, open(filename + '_params.pkl', 'wb'))
        pickle.dump(y_test, open(filename + '_ytest.pkl', 'wb'))

        y_pred = cv.best_estimator_.predict(X_test)
        pickle.dump(y_pred, open(filename + '_ypred.pkl', 'wb'))

        c+=1

if __name__ == '__main__':
    main()
