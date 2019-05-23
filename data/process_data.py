import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loads the message databases and the categories database provided in messages_filepath, categories_filepath and returns a single database
    with columns: id,message,original, and the types of disasters
    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories,on="id")
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=False, n=1).str.get(1)
    
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
    
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df,categories),axis=1)
    return(df)


def clean_data(df):
    '''
    cleans the dataframe
    '''
    
    # check number of duplicates
    duplicates=df[df.duplicated()]
    # drop duplicates
    df=df.drop(index=duplicates.index)
    
    # removes values outside of {0,1}
    columns=df.columns[4:]
    for col in columns:
        df.loc[df[col]>1,col]=1
        df.loc[df[col]<0,col]=0
    
    return(df)


def save_data(df, database_filename):
    '''
    saves the dataframe df into the sql database database_filename
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
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