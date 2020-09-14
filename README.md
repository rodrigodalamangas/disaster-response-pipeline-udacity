# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [Screenshots](#screenshots)
3. [Getting Started](#getting_started)
   - [Dependencies](#dependencies)
   - [Installing](#installation)
   - [Executing Program](#execution)
   - [Important Files](#importantfiles)
4. [Data Observations](#data)
5. [Author](#author)
6. [License](#license)

<a name="motivation"></a>
## Project Motivation

This Project is part of Data Scientist Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains messages from real-life disaster events. The main objective of the project is to build a Machine Learning model to categorize messages in real time.

This project is divided in the following key sections:

- Build an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB;
- Build a machine learning pipeline to classify text message in various categories;
- Run a web app which can show model results in real time.

<a name="screenshots"></a>
## Screenshots

![Main Page](screenshots/main.png)

![Input Page](screenshots/input.png)

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies

- Python 3
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle, Joblib
- Web App and Data Visualization: Flask, Plotly

<a name="installation"></a>
### Installing

Clone the git repository:

```
git clone https://github.com/rodrigodalamangas/disaster-response-pipeline-udacity.git
```

<a name="execution"></a>
### Executing Program:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="importantfiles"></a>
### Important Files

**app/templates/\***: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

**app/run.py**: This file can be used to launch the Flask web app used to classify disaster messages

**ETL Preparation Notebook**: learn everything about the implemented ETL pipeline

**ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

<a name="data"></a>
## Data Observations

Accordingly to data visualization, most of the classes (categories) are highly imbalanced. This affects model F1 prediction score. One using this project should take this into consideration and apply measures model selection and parameters fine-tuning to build the Machine Learning model.

<a name="author"></a>
## Author

- [Rodrigo Dalamangas](https://github.com/rodrigodalamangas)

<a name="license"></a>
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
