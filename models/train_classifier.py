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


def tokenize(text):
    """
    tokenizes the messagages into words (lowercase, remove punctuation, split, remove stopwords and lemmatize with WordNet)
    Inputs:
    - text: string
    :return: lemmed text

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


def main():
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
              'train_classifier_perf_all.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()