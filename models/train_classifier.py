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

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

import sklearn.metrics as met
from sklearn.utils.fixes import signature
import pickle

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
    pass


def save_model(model, model_filepath):
    pass


def mo_confusion_matrix(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :return:

    """
    cm = np.zeros((2, 2, y_test.shape[1])).astype(int)

    c = 0
    for column in y_test.columns:
        cm[:, :, c] = met.confusion_matrix(y_test[column], y_pred[:, c])
        c += 1
    tn = cm[0, 0, :].ravel()
    fp = cm[0, 1, :].ravel()
    fn = cm[1, 0, :].ravel()
    tp = cm[1, 1, :].ravel()

    return (tn, fp, fn, tp)

def mo_weighted_scorer(y_test, y_pred, weights={'tp': 1, 'tn': 1, 'fn': 1, 'fp': 1}, class_weights=None,
                       adjust_for_frequency=False,
                       return_single=True):
    """

    :param y_test:
    :param y_pred:
    :param weights:
    :param class_weights:
    :param adjust_for_frequency:
    :param return_sum:
    :return:
    """
    # score with custom weights for errors

    tn, fp, fn, tp = mo_confusion_matrix(y_test, y_pred)
    score = tn * weights['tn'] + tp * weights['tp'] - fn * weights['fn'] - fp * weights['tp']

    if adjust_for_frequency:
        # adjusting the weights by the frequency

        real_disasters = tp + fn
        class_inv_frequencies = 1.0 / real_disasters
        class_inv_frequencies[real_disasters == 0] = 0
        # normalize the inv. frequencies
        class_inv_frequencies /= class_inv_frequencies.max()
        #         print()
        #         print(class_inv_frequencies)
        #         print()

        if not class_weights is None:
            class_weights *= class_inv_frequencies
        else:
            # the score is just adjusted by the inverse of frequency
            class_weights = class_inv_frequencies

    if not class_weights is None:
        # adjust score by weights
        score *= class_weights

    if return_single:
        #return np.sum(np.power(score.values,2.)))
        return np.sum(score.values)
    else:
        return score


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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

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


    pipelines={'RF':[pipe_RandomForest,par_RandomForest] , 'DT':[pipe_DecisionTree,par_DecisionTree]}
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
    class_weights['death']=5
    class_weights['floods']=2
    class_weights['fire']=4
    class_weights['storm']=2
    class_weights['earthquake']=5
    class_weights['search_and_rescue']=3


    scorer = make_scorer(mo_weighted_scorer,greater_is_better = True ,weights=weights,
                         class_weights=class_weights,adjust_for_frequency=True,
                         return_single=True)

    for classifier in pipelines.keys():
        print(classifier)

        model = pipelines[classifier][0]
        parameters = pipelines[classifier][1]
        cv_folds=5


        #cv = GridSearchCV(model, param_grid=parameters, verbose=2, cv=cv_folds)
        cv = GridSearchCV(model, scoring=scorer, param_grid=parameters, verbose=2, cv=cv_folds)
        cv.fit(X_train, y_train)

        #print(cv)

        print()
        print('training finished')
        print()

        filename='%s_%s' % ('04_0.2_w',classifier)
        pickle.dump(cv.best_estimator_, open(filename + '_model.pkl', 'wb'))
        pickle.dump(cv.best_params_, open(filename + '_params.pkl', 'wb'))
        pickle.dump(y_test, open(filename + '_ytest.pkl', 'wb'))
        
        y_pred = cv.best_estimator_.predict(X_test)
        pickle.dump(y_pred, open(filename + '_ypred.pkl', 'wb'))



if __name__ == '__main__':
    main()
