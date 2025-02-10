
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

from src.utils import save_object


@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.datatransformationconfig = datatransformationconfig()

    def get_data_transfer_obj(self):
            """
            This functions defines all the steps of transformation we want to implement
            on the data. So, overall we have divided transformation file into 2 things,
            get_data_transfer_obj(self) --> Defining all steps of transformation
            initiate_data_transformation() --> Executing all the steps of transformation 
            
                'ID', 'Delivery_person_ID', 'Delivery_person_Age',
            'Delivery_person_Ratings', 'Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Order_Date', 'Time_Orderd',
            'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density',
            'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
            'multiple_deliveries', 'Festival', 'City', 'Time_taken (min)'],
            dtype='object'

                  """
            try:
                numeric_features = ['Delivery_person_Age', 
                                    'Delivery_person_Ratings', 'Restaurant_latitude','Restaurant_longitude' ,
                                    'Delivery_location_latitude', 'Delivery_location_longitude', 
                                    'Vehicle_condition', 'multiple_deliveries']
                categorical_features = ['Weather_conditions', 'Road_traffic_density', 
                                        'Type_of_order', 'Type_of_vehicle', 'Festival', 
                                        'City']
                
            # Now, we will define the kind of transformation we want to implement for numerical pipeline 
            # in a single pipeline e.g. Imputing, encoding or we can define function above and include its
            # execution in the pipeline using FunctionTransformer for example def remove_outliers().


                num_pipeline = Pipeline( 
                    steps=[            

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

                logging.info("Numerical columns standard scaling completed")
                logging.info("categorical columns encodinng completed")

                preprocessor = ColumnTransformer(
                    [
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                    
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

                target_column_name = 'Time_taken'

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