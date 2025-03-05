
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
import re 

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
            
            Failure_columns = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E']
            X['failure_type'] = X[Failure_columns].idxmax(axis=1)  # We extract the column name for which we have highest value among these columns
            X['failure_type'] = X['failure_type'].str.replace('Failure ', '', regex = True)
            X = X.drop(columns= Failure_columns)

            label_mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

            X['failure_type'] = X['failure_type'].map(label_mapping)

            #X[f'{self.model_column}_year'] =  X[self.model_column].str.extract(r'(\d{4})').astype(float)
            extracted_year = X[self.model_column].str.extract(r'(\d{4})')[0]  # Extract the year as a Series
            X[f'{self.model_column}_year'] = pd.to_numeric(extracted_year, errors='coerce')
            current_year = float(datetime.now().year)
            X['Age_of_car'] = current_year - X[f'{self.model_column}_year']

            usage_map = {
                'Medium': 20000,
                'Low': 10000,
                'High': 30000
            }

            X['Mileage'] = X[self.usage_column].map(usage_map) * X[f'{self.model_column}_year']

            if 'Temperature' in X.columns:
                 X['Temperature'] = X['Temperature'].str.replace(' °C', '').str.strip()
                 X['Temperature'] = pd.to_numeric(X['Temperature'], errors = 'coerce')   # Transformed Temperature column 

            numeric_columns = ['Temperature', 'RPM', 'Fuel consumption', 'Mileage', 'Age_of_car']
            for col in numeric_columns:
                if col in X.columns:  # Check if the column exists in the DataFrame
                    X[col] = pd.to_numeric(X[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

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

            extracted_year = X[self.date_column].str.extract(r'(\d{4})')[0]  # Extract the year as a Series
            X[f'{self.date_column}_year'] = pd.to_numeric(extracted_year, errors='coerce')

            
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
            
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.empty:
                raise ValueError("No numeric columns found for outlier removal.")

            mask = (X_numeric >= self.lower_limit) | (X_numeric <= self.upper_limit)
            Outlier_free = X[mask.all(axis=1)]
            if len(Outlier_free) == 0:
                 
                return X
            else: 
                return Outlier_free
        
            
        
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
                                    'RPM','Fuel consumption', 'Mileage', 'Age_of_car']
                categorical_features = ['Usage', 'Color', 'Factory', 
                                        'Membership'
                                    ]
                date_column = 'Model'
                
                
            # Now, we will define the kind of transformation we want to implement for numerical pipeline 
            # in a single pipeline e.g. Imputing, encoding or we can define function above and include its
            # execution in the pipeline using FunctionTransformer for example def remove_outliers().


                num_pipeline = Pipeline( 
                    steps=[            
                        ("outlier_remover", outlier_remover(factor=1.5)),
                        ("imputer", SimpleImputer(strategy='median')), 
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
                    ("date_pipeline", date_pipeline, [date_column])
                    
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

                target_column_name = 'failure_type'
                
                #target_column_name = ['Failure A','Failure B','Failure C','Failure D','Failure E'] 

                #missing_cols = [col for col in target_column_name if col not in train_df.columns]
                #if missing_cols:
                    #raise ValueError(f"Columns missing in train_df: {missing_cols}")
                
                """
                Now, we will ensure that newly created columns are part of test and training dataset.
                """

                Calculated_columns = feature_enngineering()
                train_df = Calculated_columns.fit_transform(train_df)
                test_df = Calculated_columns.transform(test_df)

                input_features_train_df = train_df.drop(columns=target_column_name, axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_features_test_df = test_df.drop(columns=target_column_name, axis=1)
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
            
