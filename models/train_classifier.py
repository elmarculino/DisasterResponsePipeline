import sys
import re
import pickle
import warnings

import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)

nltk.download(['punkt','wordnet','stopwords'])

random_state = 99


def load_data(database_filepath):
    '''
    Load training data from SQLite database
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df['message'] 
    y = df.iloc[:,4:]
    labels = y.columns.tolist()
    
    return X, y, labels
    
    
def tokenize(text):
    '''
    Return the cleaned, lemmatized, lowercased list of tokens from the text input 
    '''
    lemmatizer = WordNetLemmatizer()    
    text = re.sub(r"[^a-zA-Z]", " ", text.lower()) 
    tokens = word_tokenize(text)
    
    return [lemmatizer.lemmatize(t).strip() for t in tokens if t not in stopwords.words("english")]


def build_model():
    '''
    Build an optimized machine learning model using scikit-learn Pipeline and GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier(random_state=random_state)))
    ])
    
    parameters = {
        'moc__estimator__n_estimators': [ 50, 100 ],
        'moc__estimator__max_depth': [ 3, 5 ]
    }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate models accuracy
    '''
    Y_preds = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_preds, target_names=category_names))
    

def save_model(model, model_filepath):
    '''
    Save model into a pickle file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

        
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()