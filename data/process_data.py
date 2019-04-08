import sys
import pandas as pd
import numpy as np
import pylab as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages_filepath and categories_filepath and merges them into the df DataFrame
    :param messages_filepath:
    :param categories_filepath:
    :return:
    """

    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    df = messages.merge(categories, on="id")

    return(df)

def clean_data(df):
    """
    cleans the DataFrame df (removes duplicates etc) and returns it
    :param df: DataFrame df
    :return: DataFrame df
    """
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row]

    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=False, n=1).str.get(1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(columns=['categories'])
    df = pd.concat((df, categories), axis=1)

    # check number of duplicates
    duplicates = df[df.duplicated()]
    df = df.drop(index=duplicates.index)

    duplicates = df[df.duplicated()]

    return (df)


def save_data(df, database_filename):
    """
    saves the input DataFrame df into the SQL database database_filename
    :param df: input DataFrame
    :param database_filename: SQL database database_filename
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')

    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()