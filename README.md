# Capstone Project for Azure Machine Learning Engineer Nanodegree in Udacity 

This project is the capstone project for "Machine Learning Engineer for Microsoft Azure" Udacity's Nanodegree. In this project we will choose 
a public external dataset. This dataset will be used for train a model using 1) an Automated ML 2) Hyperdrive. After we will compare the performance of these
two different algorithms and deploy the best model. Finally the endpoint produced will be used to get some answers about predictions.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview

*TODO*: Explain about the data you are using and where you got it from.

The dataset I used is heart_failure_clinical_records_dataset.csv(https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). It describes some recorded health indicators metrics. The data have almost 300 rows of these indicators recorded from patients. 

### Task

*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

I am using this data in order to predict the DEATH_EVENT i.e. whether or not the patient deceased during the follow-up period (boolean). The features of the data are the following:

age : The age of the patient.

anaemia : Decrease of red blood cells or hemoglobin (boolean).

creatinine_phosphokinase : Decrease of red blood cells or hemoglobin (boolean).

diabetes : If the patient has diabetes (boolean).

ejection_fraction : Percentage of blood leaving the heart at each contraction (percentage).

high_blood_pressure : If the patient has hypertension (boolean).

platelets : Platelets in the blood (kiloplatelets/mL).

serum_creatinine : Level of serum creatinine in the blood (mg/dL).

serum_sodium : Level of serum sodium in the blood (mEq/L).

sex : Woman or man (binary).

smoking : If the patient smokes or not (boolean).

time : Follow-up period (days).

DEATH_EVENT : If the patient deceased during the follow-up period (boolean).


### Access
*TODO*: Explain how you are accessing the data in your workspace.

I upload the dataset in the Azure ML studio from local file (the one that I have uploaded also here in github heart_failure_clinical_records_dataset.csv). As you can see in either the automl.ipynb and hyperparameter_tuning.ipynb the code is checking whether or not the .csv has been uploaded, if not then the code makes the dataset getting it from this repo. 

## Automated ML

*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

The AutoML settings I have used are below : 

```
automl_settings = {"n_cross_validations": 2,
                    "primary_metric": 'accuracy',
                    "enable_early_stopping": True,
                    "max_concurrent_iterations": 4,
                    "experiment_timeout_minutes": 20,
                    "verbosity": logging.INFO
                    }
```      

```
automl_config = AutoMLConfig(compute_target = compute_target,
                            task='classification',
                            training_data=dataset,
                            label_column_name='DEATH_EVENT',
                            path = project_folder,
                            featurization= 'auto',
                            debug_log = "automl_errors.log",
                            enable_onnx_compatible_models=False,
                            blocked_models = ['XGBoostClassifier'],
                            **automl_settings
                            )
```

* ```n_cross_validations``` : It is how many cross validations set to make when user validation data is not specified. The main set of data is split to ```n=2``` sets and it is performed train on the one of the two and validation to the other set. So this procedure is performed two times, because we have ```n_cross_validations=2```. 

* ```primary_metric = 'accuracy' ``` :  The metric that Automated Machine Learning will optimize for model selection. We have set the 'accuracy'.

* ``` enable_early_stopping = True``` : Whether to enable early termination if the score is not improving in the short term. 

* ``` max_concurrent_iterations = 4``` : The maximum number of iterations that could be executed in parallel.  It is recommended you create a dedicated cluster per experiment, and match the number of max_concurrent_iterations of your experiment to the number of nodes in the cluster. This way, you use all the nodes of the cluster at the same time with the number of concurrent child runs/iterations you want. For this I set it to 4.

* ``` experiment_timeout_minutes = 20 ``` :  It defines how long, in minutes, your experiment should continue to run. In previous projects we couldn't set more than 30 minutes. In this project we could use more but it is not needed for so small training set. However, it is for sure something that you could change for better performance.

* ``` "verbosity": logging.INFO ``` : It is the verbosity level you want to produced.

* ``` compute_target = compute_target``` : The compute target with specific vm_size and max_nodes. The one that has been configured with name 'aml_compute' in the automl.ipynb.

* ``` task='classification' ``` : We have a classification task to do, we seek to predict whether or not the person will have a heart failure. With other words we are trying to predict the ``` DEATH_EVENT ```.

* ``` training_data = dataset ``` : The data on which the algorithm will be trained.

* ``` label_column_name='DEATH_EVENT' ``` : The name of the column that contains the labels of the train data, i.e the target column we want to predict.

* ``` path= project_folder``` : The path to the Azure ML folder of the project.

* ``` featurization= 'auto' ``` : Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used. I used ``` auto``` so featurization step step should be done automatically.

* ``` debug_log = "automl_errors.log" ``` : The debug information are written to the  ```automl_errors.log```.

* ``` enable_onnx_compatible_models = False ``` : Whether to enable or disable enforcing the ONNX-compatible models.

* ``` blocked_models = ['XGBoostClassifier'] ``` : What algorithm we want from AutoML to not run. I selected ``` XGBoostClassifier```, the answer could be found in here https://knowledge.udacity.com/questions/509841. For those who don't have access it is for compatibility issues. So the lack of time to make the ``` XGBoostClassifier``` to run make me to enforce the AutoML to not run this specific algorithm. 


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
