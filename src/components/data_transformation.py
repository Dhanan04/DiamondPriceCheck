from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass

from src.utils import save_object

@dataclass 

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts' , 'preprocessor.pkl')
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
           
           ## categorical and numerical columns
            
            Numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
            Categorical_Columns = ['cut', 'color', 'clarity']
            
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Data transformation Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,Numerical_columns),
                ('cat_pipeline',cat_pipeline,Categorical_Columns)
            ])
            
            logging.info('DataTransformation Completed')
            
            return preprocessor

        
        except Exception as e:
            logging.info('Exception occured in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformtion(self , train_data_path , test_data_path):
        
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            
            logging.info('Read train and test data completed')
            logging.info(f'Train dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column = 'price'
            drop_column=[target_column,'id']  
            
            ## dividing the data into dependent and independent features
            ## for train data
            input_feature_train_df = train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df = train_df[target_column]
            
            ## for test data independent and dependent feature
            input_feature_test_df = test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df = test_df[target_column]
            
            ##Data transformation
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
                
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                    
            )
            
            
            logging.info("Applying preprocessing object on test and training datasets completed")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        
        
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        