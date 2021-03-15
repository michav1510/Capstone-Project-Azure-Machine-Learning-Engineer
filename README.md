# Capstone Project for Azure Machine Learning Engineer Nanodegree in Udacity 

This project is the capstone project for "Machine Learning Engineer for Microsoft Azure" Udacity's Nanodegree. In this project we will choose 
a public external dataset. This dataset will be used for train a model using 1) an Automated ML 2) Hyperdrive. After we will compare the performance of these
two different algorithms and deploy the best model. Finally the endpoint produced will be used to get some answers about predictions.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview

*TODO*: Explain about the data you are using and where you got it from.

The dataset I used is [heart_failure_clinical_records_dataset.csv](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). It describes some recorded health indicators metrics. The data have almost 300 rows of these indicators recorded from patients. 

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

* ``` blocked_models = ['XGBoostClassifier'] ``` : What algorithm we want from AutoML to not run. I selected ``` XGBoostClassifier```, the answer could be found in the 
[forum](https://knowledge.udacity.com/questions/509841). For those who don't have access, I have to say that it is for compatibility issues. So the lack of time to make the ``` XGBoostClassifier``` to run make me to enforce the AutoML to not run this specific algorithm. 


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

In the below screenshots you can verify that the best run is the one at the below table: 
Best algorightm | VotingEnsemble
--- | ---  
Run Id | AutoML_c4027857-1690-446a-959e-02880c59deed_49
Accuracy | 0.85622
AUC macro | 0.89710
AUC micro | 0.90959


![automl_best_run_1](/Screenshots/automl_best_run_1.png)
![automl_best_run_2](/Screenshots/automl_best_run_2.png)
![automl_best_run_3](/Screenshots/automl_best_run_3.png)

I think that there is a lot of space to improve the above results. Some improvements could be : 
1) Set bigger ```n_cross_validation``` equals to 5 or 7. I think 2 sets are somehow small. 
2) I would let the algorithm to run for much more time. Thus I would change  ``` "experiment_timeout_minutes": 20 ``` to an hour or more. 
3) I would somehow enables also the ``` XGBoostClassifier ``` to run and fix the compatibility issues.
4) I would disable the early stopping policy and thus set ``` enable_early_stopping = False```.


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with its parameters.

In the following screenshots we can see the `RunDetails` widget : 
![run_details_widget_automl_1](/Screenshots/run_details_widget_automl_1.png)
![run_details_widget_automl_2](/Screenshots/run_details_widget_automl_2.png)
![run_details_widget_automl_3](/Screenshots/run_details_widget_automl_3.png)




## Hyperparameter Tuning

*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Tha algorithm used for the training is Logistic Regression. The two hyperparameters of the Logistic Regression are tuned with the hyperdrive to find the model with the best accuracy on the test set. The two hyperparameters are the following:

``` C ``` : The inverse of the reqularization strength. The smaller the number the stronger the regularization.

``` max_iter ``` : Maximum number of iterations to converge.

**Benefits of the parameter sampler**

I chose the ```RandomParameterSampling```, the hyperparameters are randomly selected from the search space. The search space for the two hyperaparameters is the following:

```
   '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
   '--max_iter': choice(50,100,300)
```
where the choice define discrete space over the values. The benefits of the ```RandomParameterSampling```, is that it is more fast than for example the ```GridParameterSampling``` where all the possible values from the search space are used, and it supports early termination of low-performance runs.

I chose the ```BanditPolicy``` which is an "aggressive" early stopping policy with the meaning that cuts more runs than a conservative one like the ```MedianStoppingPolicy```, so it saves computational time. There are three configuration parameters ```slack_factor, evaluation_interval(optional), delay_evaluation(optional)```. 
* ```slack_factor/slack_amount``` : (factor)The slack allowed with respect to the best performing training run.(amount) Specifies the allowable slack as an absolute amount, instead of a ratio.

* ```evaluation_interval``` : (optional) The frequency for applying the policy.

* ```delay_evaluation``` : (optional) Delays the first policy evaluation for a specified number of intervals.

I set ```evaluation_interval=2, slack_factor=0.1```.

About the other parameters of the ```HyperDriveConfig``` : 

* ```primary_metric_name='Accuracy'``` : Meaning that our interest is about the 'Accuracy'. 
                                    

* ```primary_metric_goal=PrimaryMetricGoal.MAXIMIZE```: Meaning that we want to Maximize the primary metric and not minimize it for example.

* ```policy=early_termination_policy``` : This is where I have set the policy I described above.

* ```max_concurrent_runs=4 ``` : Maximum number of runs to run concurrently. Of course the best is 4 as it is the number of cluster nodes.

* ```max_total_runs=16``` : Maximum number of runs. This is the upper bound, there may be fewer runs when the sample space is smaller than this value. 

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

In the below screenshots you can verify that the best run is the one at the below table: 
runId | HD_d145f066-c650-4d4a-877b-0b85cc305c65_0
--- | ---  
 Accuracy | 0.778 
 Regularization Strength  | 500
 Max Iterations | 50
 
![best_run_hyperdrive_model_1.png](/Screenshots/best_run_hyperdrive_model_1.png)
![best_run_hyperdrive_model_2.png](/Screenshots/best_run_hyperdrive_model_2.png)
![best_run_hyperdrive_model_3.png](/Screenshots/best_run_hyperdrive_model_3.png)

We can understand that there is a lot of difference from the perspective of accuracy between the Hyperdrive and the AutoML since the last hit an accuracy of ```0.85622``` and the hyperdive's accuracy is ```0.778```. Some improvements for the Hypderdrive method could be :

1) Set a ```GridParameterSampling``` so we can check more hyperparameters for better results.
2) Change the policy to a more "conservative" one. I would set as policy the ```MedianStoppingPolicy``` so the runs wouldn't stop more easy.
3) Set as ```max_total_runs``` a much more big number.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

In the following screenshots we can see the `RunDetails` widget : 
![run_details_widget_automl_1.png](/Screenshots/run_details_widget_automl_1.png)
![run_details_widget_automl_2.png](/Screenshots/run_details_widget_automl_2.png)
![run_details_widget_automl_3.png](/Screenshots/run_details_widget_automl_3.png)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
 
