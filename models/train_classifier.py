# import libraries
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import sys
import re
import pickle
import time
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


stop_words = stopwords.words("english")


def load_data(database_filepath):
    """
    Load Data from the Database Function

    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """

    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql('SELECT * FROM df', con=engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text function

    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        tokens -> List of tokens extracted from the provided text
    """

    #stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word, pos='v')
              for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build a Machine Learning Model Pipeline function

    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.

    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier())),
    ])

    parameters = {'clf__estimator__learning_rate': [0.5, 0.75, 1],
                  'clf__estimator__n_estimators': [25, 50, 75, 100]}

    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='f1_micro', n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function

    This function applies a ML pipeline to a test set and prints out the model performance (f1score)

    Arguments:
        model -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save model function

    This function saves trained model as Pickle file, to be loaded later.

    Arguments:
        model -> GridSearchCV
        model_filepath -> destination path to save .pkl file

    """

    outfile = open(model_filepath, 'wb')
    pickle.dump(model.best_estimator_, outfile)
    outfile.close()


def main():
    """
    Train Classifier Main function

    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle file

    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
