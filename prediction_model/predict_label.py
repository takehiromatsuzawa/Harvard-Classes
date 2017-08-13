
from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import json
import pandas as pd
import time
from operator import itemgetter
import sys

scopes = ['https://www.googleapis.com/auth/prediction', 'https://www.googleapis.com/auth/devstorage.full_control', \
          'https://www.googleapis.com/auth/cloud-platform']

# python predict_label.py model_name project_id credential_file_name input_file_name output_file_name
# Example: python predict_label.py job-identifier-1-2 data-science-153816 Data-Science-6af3c695f128.json test_groove.csv output.csv

# @model_name: model created in create_model.py
model_name           = sys.argv[1]

# @project_id: reference: https://support.google.com/cloud/answer/6158840?hl=en
project_id         = sys.argv[2]

# @credential_file_name: reference: http://gspread.readthedocs.io/en/latest/oauth2.html
credential_file_name = sys.argv[3]

# @input_file: file whose label you want to predict
input_file           = sys.argv[4]

# @predict_file: output file that contains predicted labels
predict_file         = sys.argv[5]


credentials = ServiceAccountCredentials.from_json_keyfile_name(credential_file_name, scopes=scopes)
http_auth = credentials.authorize(Http())
service = build('prediction', 'v1.6', http = http_auth)
papi = service.trainedmodels() 


def predict( project , mod , val ):
    body = {'input' : {'csvInstance': val}}
    response = papi.predict(project = project, id = mod, body = body).execute()
    return response

def predict_csv_file( input_file , predict_file , project_id , model_name ):
    # get data from csv file
    job_data = pd.read_csv(input_file)

    job_data =job_data
    # remove non-ascii characters and NA
    job_data = job_data.dropna()
    job_data['Title'] = job_data['Title'].apply(lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    # get inputs
    inputs = list(job_data['Title'])
    
    # @model_name: the model created in create_model.py
    predicted_label=[]
    # predict labels
    for ith in range(0,len(inputs)):
        time.sleep(0.100)
        print ith
        res_label=sorted(predict( project = project_id , mod = model_name  , val = [inputs[ith]])['outputMulti'] , key = itemgetter('score') , reverse = True)
        if res_label[0]['label']=='':
            predicted_label.append(res_label[1]['label'])
        else:
            predicted_label.append(res_label[0]['label'])

    output_data = pd.DataFrame(
        {'predicted_label': predicted_label,
         'job_title': inputs
        })

    output_data.to_csv( predict_file , index=False )

predict_csv_file( input_file, predict_file , project_id , model_name )

