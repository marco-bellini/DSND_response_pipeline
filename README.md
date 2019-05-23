# Disaster Response Pipeline Project

### Description
This is the deliverable for the Udacity project:Disaster Response Pipeline.
It demonstates a NLP pipeline based on the Figure Eight dataset. The challenge in this project is that disasters are very infrequent events, hence the classes are unbalanced and precision and recall are extremely important 

### Dependencies and Installation
- NTLK with wordnet and stopwords
- SQLalchemy
- Flask, Plotly

###  Files in the repository
- data (process_data.py: cleans data and stores in database)  
- models (train_classifier.py:  trains classifier and saves the model)
- app (run.py: runs the web dashboard)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## References
* [Figure Eight](https://www.figure-eight.com/) disaster messages dataset 


