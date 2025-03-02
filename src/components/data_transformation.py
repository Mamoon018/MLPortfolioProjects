
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object
from datetime import datetime


@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")


class feature_enngineering(BaseEstimator, TransformerMixin):
    def __init__(self, model_column = 'Model', usage_column = 'Usage'):
          self.model_column = model_column
          self.usage_column = usage_column
          

    def fit(self,X,y= None):
         return self 
    
    def transform(self, X, y = None):
        
        try:
        
            X = X.copy()

            X[f'{self.model_column}_year'] = pd.to_datetime(X[self.model_column]).dt.year
            current_year = datetime.now().year
            X['Age_of_car'] = current_year - X[f'{self.model_column}_year']

            usage_map = {
                'Medium': 20000,
                'Low': 10000,
                'High': 30000
            }

            X['Mileage'] = X[self.usage_column].map(usage_map) * X[f'{self.model_column}_year']

            logging.info("Feature engineering done")

            return X
        except Exception as e:
                raise CustomException(e,sys)


class DateSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        logging.info("splitting date")
        try:
            X = X.copy()
            date_series = pd.to_datetime(X[self.date_column])
            X[f'{self.date_column}_year'] = date_series.dt.year
            X[f'{self.date_column}_month'] = date_series.dt.month
            X.drop(columns=[self.date_column], inplace=True)
            return X

        except Exception as e:
                raise CustomException(e,sys)

class outlier_remover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
         
         """
         we will start with initializing the transformer.
         
         """
         self.factor = factor
         self.lower_limit = None 
         self.upper_limit = None 

    def fit(self, X, y=None):
         """
         we will give fit the formula to calculate 
         IQR for each numeric column so, that when 
         we run .fit_transform,
         """

         Q1 = np.percentile(X,25, axis=0)
         Q3 = np.percentile(X,75, axis=0)

         IQR = Q3 - Q1
         self.lower_limit = Q1 - self.factor * IQR 
         self.upper_limit = Q3 + self.factor * IQR
         return self 
    
    def transform(self, X, y = None):

        logging.info("removing outliers")
        try:
            if self.lower_limit is None or self.upper_limit is None:
                raise ValueError("Outlier has not been fitted yet.")
            
            
            mask = (X >= self.lower_limit) & (X <= self.upper_limit)
            return X[mask.all(axis=1)]
            
        
        except Exception as e:
                raise CustomException(e,sys)

class Datatransformation:
    def __init__(self):
        self.datatransformationconfig = datatransformationconfig()

    def get_data_transfer_obj(self):
            """
            This functions defines all the steps of transformation we want to implement
            on the data. So, overall we have divided transformation file into 2 things,
            get_data_transfer_obj(self) --> Defining all steps of transformation
            initiate_data_transformation() --> Executing all the steps of transformation

                  """
            try:

                numeric_features =  ['Temperature', 
                                    'RPM', 'Usage','Fuel consumption', 'Mileage', 'Age_of_car',
                                    ]
                categorical_features = ['Color', 'Factory', 
                                        'Membership'
                                    ]
                date_column = ['Model']
                Encoded_features = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E']
                
            # Now, we will define the kind of transformation we want to implement for numerical pipeline 
            # in a single pipeline e.g. Imputing, encoding or we can define function above and include its
            # execution in the pipeline using FunctionTransformer for example def remove_outliers().


                num_pipeline = Pipeline( 
                    steps=[            

                        ("imputer", SimpleImputer(strategy='median')),
                        ("outlier_remover", outlier_remover(factor=1.5)), 
                        ("standard_scaler", StandardScaler())
                    ]
                )
                categorical_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy= 'most_frequent')),
                        ("Encoding", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean=False))

                    ]
                )

                date_pipeline = Pipeline(
                    steps=[
                        ( "Feature_engineering", feature_enngineering()),
                        ( "date_splitter", DateSplitter(date_column=date_column)), 
                        ("standard_scaler", StandardScaler())
                    ]
                )


                logging.info("Numerical columns standard scaling completed")
                logging.info("categorical columns encodinng completed")

                preprocessor = ColumnTransformer(
                    [
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features),
                    ("date_pipeline", date_pipeline, date_column),
                    ("Encoded_pass", "passthrough", Encoded_features)
                    
                    ],
                    remainder="drop"
                )

                return preprocessor
            except Exception as e:
                raise CustomException(e,sys)
                
    def initiate_data_transformation(self, train_path, test_path):
            
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                

                logging.info("Read data from train and test completed")
                logging.info("obtaining preprocessing object")

                preprocessing_obj = self.get_data_transfer_obj()

                target_column_name = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E'] 

                input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessibg on training and test dataframe")

                input_feature_train_arry=preprocessing_obj.fit_transform(input_features_train_df)
                # When we apply fit_transform, it gets for imputing, scaler, encoding parameters,
                # and stores them all in the preprocessing object (only parameters, not data). So, now these learned 
                # parameters are stored in the individual pipelines that are part of the column transformer. 
                # IMP POINT: now, if we save preprocessor in artifact file, and whenever we will call preprocessor
                # it will come with learned parameters. And we will just need to apply transform on the new data
                # to transform it using learned parameters.
                #Example:
                # January data: Fit the preprocessor (learns scaling factors, imputation values).
                # February data: Transform using the already learned parameters from January, if use saved file.

                input_feature_tets_arry=preprocessing_obj.transform(input_features_test_df)

                train_arr = np.c_[
                        input_feature_train_arry, np.array(target_feature_train_df)
                                
                                 ]
                test_arr = np.c_[
                        input_feature_tets_arry, np.array(target_feature_test_df)]

                logging.info(f" Saveing preprocessor object.")

                save_object(

                    file_path = self.datatransformationconfig.preprocessor_obj_file_path,
                    obj = preprocessing_obj
                # It will save the preprocessor object with the learned parameters.
                ) 

                return ( 

                    train_arr,
                    test_arr,
                    self.datatransformationconfig.preprocessor_obj_file_path
                
                )             

            except Exception as e:
                raise CustomException(e,sys)
            
