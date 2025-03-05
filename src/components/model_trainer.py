
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

            model_report:dict= evaluate_models(x_train = X_train, y_train= Y_train, x_test=X_test, y_test=Y_test, 
                                              models=ML_models)
            
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