import pandas as pd
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing

# import modules for api requests
import requests

# import imputing function
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#******************************************** Acquire Titanic Data **************************************************#

def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df

def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to a csv file if a local file does not exist, and returns a df.
    ''' 
    if os.path.isfile('titanic.csv'):
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic.csv', index_col=0)
        
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv('titanic.csv')

        # Return the dataframe to the calling code
    return df

#******************************************** Acquire Iris Data **************************************************#

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT *
                FROM measurements
                JOIN species USING(species_id);
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df


def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df


#************************************************ Acquire Telco Data **************************************************#


def new_telco_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        return df
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df

#******************************************** Acquire and Wrangle Zillow Data **************************************************#
# This function connects to the Codeup database.
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
# This function fetches the specified data and stores it as a Pandas dataframe
def zillow_sql():
    '''
    This function reads the zillow data from the Codeup database into a Pandas dataframe.
    '''
    sql_query = """
                SELECT * FROM predictions_2017
                left join properties_2017 using(parcelid)
                where properties_2017.propertylandusetypeid = 261;
                """
    
    # Read in DataFrame from Codeup db.
    zillow_df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return zillow_df

# This function caches the zillow data.

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        zillow_df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        zillow_df = zillow_sql()
        
        # Cache data
        zillow_df.to_csv('zillow.csv')
        
    return zillow_df


#******************************************** Acquire Data from https://python.zgulde.net**************************************************#

def new_retail_data(base_url='https://python.zgulde.net'):
    '''
    This function acquires new retail data, returns three dataframes, and saves those dataframes to .csv files.
    '''
    # Acquiring items data
    response = requests.get('https://python.zgulde.net/api/v1/items')
    data = response.json()
    df = pd.DataFrame(data['payload']['items'])
    while data['payload']['next_page'] != None:
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        df = pd.concat([df, pd.DataFrame(data['payload']['items'])]).reset_index(drop=True)
    items_df = df.copy()
    print("Items data acquired...")

    # Acquiring stores data
    response = requests.get('https://python.zgulde.net/api/v1/stores')
    data = response.json()
    df = pd.DataFrame(data['payload']['stores'])
    stores_df = df.copy()
    print("Stores data acquired...")

    # Acquiring sales data
    response = requests.get('https://python.zgulde.net/api/v1/sales')
    data = response.json()
    df = pd.DataFrame(data['payload']['sales'])
    while data['payload']['next_page'] != None:
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        df = pd.concat([df, pd.DataFrame(data['payload']['sales'])]).reset_index(drop=True)
    sales_df = df.copy()
    print("Sales data acquired")

    # Saving new data to .csv files
    items_df.to_csv("items.csv", index=False)
    stores_df.to_csv("stores.csv", index=False)
    sales_df.to_csv("sales.csv", index=False)
    print("Saving data to .csv files")

    return items_df, stores_df, sales_df

def get_store_data():
    '''
    This function reads in retail data from the website if there are no csv files to pull from
    '''
    # Checks if .csv files are present. If any are missing, will acquire new data for all three datasets
    if (os.path.isfile('items.csv') == False) or (os.path.isfile('sales.csv') == False) or (os.path.isfile('stores.csv') == False):
        print("Data is not cached. Acquiring new data...")
        items_df, stores_df, sales_df = new_retail_data()
    else:
        print("Data is cached. Reading from .csv files")
        items_df = pd.read_csv('items.csv')
        print("Items data acquired...")
        stores_df = pd.read_csv('stores.csv')
        print("Stores data acquired...")
        sales_df = pd.read_csv('sales.csv')
        print("Sales data acquired...")

    combined_df = sales_df.merge(items_df, how='left', left_on='item', right_on='item_id').drop(columns=['item'])
    combined_df = combined_df.merge(stores_df, how='left', left_on='store', right_on='store_id').drop(columns=['store'])
    print("Acquisition complete")
    return combined_df

def new_power_data():
    opsd = pd.read_csv("https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv")
    opsd = opsd.fillna(0)
    print("Saving data to .csv file")
    opsd.to_csv('opsd_germany_daily_data.csv', index=False)
    return opsd

def get_power_data():
    if os.path.isfile('opsd_germany_daily_data.csv') == False:
        print("Data is not cached. Acquiring new power data.")
        opsd = new_power_data()
    else:
        print("Data is cached. Reading data from .csv file.")
        opsd = pd.read_csv('opsd_germany_daily_data.csv')
    print("Acquisition complete")
    return opsd