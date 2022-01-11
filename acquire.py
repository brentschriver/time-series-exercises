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

def get_all(endpoint):
    """ Read all records on all pages """
    
    if endpoint not in ["sales", "items", "stores"]:
        return "Not available from this API. Check the documentation"
    
    host = "https://python.zgulde.net/"
    api = "api/v1/"

    url = host + api + endpoint

    response = requests.get(url)

    if response.ok:
        payload = response.json()["payload"]

        # endpoint should be "items", "sales", or "stores"
        contents = payload[endpoint]

        # Make a dataframe of the contents
        df = pd.DataFrame(contents)

        next_page = payload["next_page"]

        while next_page:
            # Append the next_page url piece
            url = host + next_page
            response = requests.get(url)

            payload = response.json()["payload"]

            next_page = payload["next_page"]    
            contents = payload[endpoint]

            df = pd.concat([df, pd.DataFrame(contents)])

            df = df.reset_index(drop=True)

    return df