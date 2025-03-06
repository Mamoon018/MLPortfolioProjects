
import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostClassifier 
from sklearn.ensemble import (
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier 

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

## We will start with config file and specify the path of file

@dataclass
class ModelTrainerConfig:
    trainer_model_configfile = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiating_model_trainer(self,train_array, test_array):  # These two parameters refer to the output of the data transformation.
        try:
            logging.info("Splitting training and test input data")

            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1], # All all columns of train_array except last column which is our target column.
                train_array[:,-1], # only last column extracted
                test_array[:,:-1],
                test_array[:,-1]

            )

            ML_models = {

                
                "K-Neighbours Classifier": KNeighborsClassifier(),
                "XGB Classifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "Adaboost Classifier": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()

            }

            tuning_parameters = {

                "K-Neighbours Classifier": {
                    'n_neighbors': [3,5,7,10,15],
                    #'weights': ['uniform', 'distance'],
                    #'p': [1,2]

                },
            
                "XGB Classifier": {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    #'max_depth': [3,5,7,10],
                    #'subsample':[0.6,0.7,0.8,0.9,1.0],
                    #'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
                    #'objective': ['binary:logistic']

                },

                "CatBoosting Classifier": {

                    'iterations': [30,50,100],
                    'learning_rate': [0.01,0.05,0.1],
                    #'depth':[6,8,10],
                    #'l2_leaf_reg':[1,3,5,7,9],
                    #'border_count':[32,64,128]
                },

                "Adaboost Classifier":{

                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.1,0.01,0.5,0.001],
                    
                },

                "Gradient Boosting": {

                    #'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.1,0.01, 0.05,0.001],
                    #'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    #'max_depth': [3,5,7,10],
                    #'min_samples_leaf': [1,2,4]
                },

                "Random Forest": {

                    'n_estimators': [8,16,32,64,128,256],
                    #'criterion': ['gini', 'entropy'],
                    #'max_depth': [None, 10, 20, 30,40, 50],
                    #'min_samples_split': [2,5,10],
                    #'min_samples_leaf': [1,2,4],
                    #'max_features': ['sqrt','log2', None]
                },

                "Decision Tree": {


                    'criterion': ['gini','entropy'],
                    #'splitter': ['best', 'random'],
                    #'max_depth': [None, 10,20,30,40,50],
                    #'min_samples_split': [2,5,10],
                    #'min_samples_leaf': [1,2,4],
                    #'max_features': ['sqrt', 'log2', None]
                }

            }


            model_report:dict= evaluate_models(x_train = X_train, y_train= Y_train, x_test=X_test, y_test=Y_test, 
                                              models=ML_models, param= tuning_parameters )
            
            ## We will extract the best model score from report

            Best_model_score = max(sorted(model_report.values()))

            ## Best model name

            Best_model_name = list(model_report.keys())[
                list(model_report.values()).index(Best_model_score)
                                                        ]
            best_model = ML_models[Best_model_name]

            if Best_model_score < 0.6:
                raise CustomException("Not a single model model performed well")
            logging.info(f"Best model {Best_model_name} with score {Best_model_score} found on training and test dataset")

            save_object(

                file_path = self.model_trainer_config.trainer_model_configfile,
                obj = best_model                 
            )


            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(Y_test,predicted)
            return accuracy, Best_model_name, Best_model_score, best_model, model_report

        except Exception as e:
            raise CustomException(e,sys)
        
