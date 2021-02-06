# Capstone Project for Azure Machine Learning Engineer Nanodegree in Udacity 

This project is the capstone project for "Machine Learning Engineer for Microsoft Azure" Udacity's Nanodegree. In this project we will choose 
a public external dataset. This dataset will be used for train a model using 1) an Automated ML 2) Hyperdrive. After we will compare the performance of these
two different algorithms and deploy the best model. Finally the endpoint produced will be used to get some answers about predictions.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The dataset I used is weatherAUS.csv(https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). It describes some recorded weather data for years 2008,2009,2010 from
some places in Australia. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
I am using this data in order to predict the rainfall of a given date, location and temperature data. The features of the data are the following: 

Date: The given date of which the rainfall happened or not.

Location : The town/state in which the data recorded.

MinTemp : The minimum temperature recorded for the given date and location.

MinTemp : The maximum temperature recorded for the given date and location.

RainFall : The amount of rainfall recorded for the day in mm.

Evaporation : The so-called Class A pan evaporation (mm) in the 24 hours to 9am.

Sunshine : The number of hours of bright sunshine in the day.

WindGustDir : The direction of the strongest wind gust in the 24 hours to midnight.

WindGustSpeed : The speed (km/h) of the strongest wind gust in the 24 hours to midnight.

WindDir9am : Direction of the wind at 9am.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
