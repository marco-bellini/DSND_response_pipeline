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


#######
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

import gensim
import keras

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Activation, SpatialDropout1D
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder



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
              'train_classifier_perf_all.py ../data/DisasterResponse.db classifier.pkl')

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
        ('clf', MultiOutputClassifier(LogisticRegression(multi_class='ovr'))),
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


    pipelines={'RF':[pipe_RandomForest,par_RandomForest] , 'DT':[pipe_DecisionTree,par_DecisionTree],
               'LR':[pipe_LogisticReg,par_LogisticReg],'MN':[pipe_MultinomialNB,par_MultinomialNB],
               'GB':[pipe_GradBoost,par_GradBoost]

               }
    return(pipelines)

def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)



def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(2))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    return dnn_model

def main():

    database_filepath = 'sqlite:///'+'..//data//InsertDatabaseName.db'
    X, Y, category_names = load_data(database_filepath)

    print('database loaded')
    print('X:', X.shape)

    X_train, X_test, yout_train, yout_test = train_test_split(X, Y, test_size=0.2)


    # encoding
    le = LabelEncoder()
    num_classes = y_train.columns.shape[0]

    # tokenize train reviews & encode train labels
    tokenized_train = [tokenize(text)  for text in X_train]

    y_tr = le.fit_transform(yout_train)
    y_train = keras.utils.to_categorical(y_tr, num_classes)
    # tokenize test reviews & encode test labels
    tokenized_test = [tn.tokenizer.tokenize(text)
                      for text in X_test]
    y_ts = le.fit_transform(yout_test)
    y_test = keras.utils.to_categorical(y_ts, num_classes)

    # build word2vec model
    w2v_num_features = 500
    w2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_features, window=150,
                                       min_count=10, sample=1e-3)

    # generate averaged word vector features from word2vec model
    avg_wv_train_features = averaged_word2vec_vectorizer(corpus=tokenized_train, model=w2v_model,
                                                         num_features=500)
    avg_wv_test_features = averaged_word2vec_vectorizer(corpus=tokenized_test, model=w2v_model,
                                                        num_features=500)

    # feature engineering with GloVe model
    train_nlp = [tn.nlp(item) for item in X_train]
    train_glove_features = np.array([item.vector for item in yout_test])

    test_nlp = [tn.nlp(item) for item in X_test]
    test_glove_features = np.array([item.vector for item in yout_test])

    print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape, ' Test features shape:',
          avg_wv_test_features.shape)
    print('GloVe model:> Train features shape:', train_glove_features.shape, ' Test features shape:',
          test_glove_features.shape)

    w2v_dnn = construct_deepnn_architecture(num_input_features=500)

    if 0:
        batch_size = 100
        w2v_dnn.fit(avg_wv_train_features, y_train, epochs=5, batch_size=batch_size,
                    shuffle=True, validation_split=0.1, verbose=1)

        y_pred = w2v_dnn.predict_classes(avg_wv_test_features)
        predictions = le.inverse_transform(y_pred)


    if 0:
        glove_dnn = construct_deepnn_architecture(num_input_features=300)

        batch_size = 100
        glove_dnn.fit(train_glove_features, y_train, epochs=5, batch_size=batch_size,
                      shuffle=True, validation_split=0.1, verbose=1)

        y_pred = glove_dnn.predict_classes(test_glove_features)
        predictions = le.inverse_transform(y_pred)





if __name__ == '__main__':
    main()
