from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import sys


scopes = ['https://www.googleapis.com/auth/prediction', 'https://www.googleapis.com/auth/devstorage.full_control', \
          'https://www.googleapis.com/auth/cloud-platform']

# python create_model.py model_name project_id credential_file_name data
# Example: python create_model.py job-identifier-1-2 'data-science-153816'  Data-Science-6af3c695f128.json 'quickstart-1483199722/job_title_gapi.csv'

# @model_name: model created in create_model.py
model_name            = sys.argv[1]

# @project_id: reference: https://support.google.com/cloud/answer/6158840?hl=en
project_id            = sys.argv[2]

# @credential_file_name: reference: http://gspread.readthedocs.io/en/latest/oauth2.html
credential_file_name  = sys.argv[3]

# @storage_name: reference: https://cloud.google.com/prediction/docs/quickstart
storage_data_location = sys.argv[4]


x_file = open( credential_file_name , "r")

credentials = ServiceAccountCredentials.from_json_keyfile_name( credential_file_name , scopes=scopes)
http_auth = credentials.authorize(Http())
service = build('prediction', 'v1.6', http=http_auth)
papi = service.trainedmodels()  

# create model
def create_model(project, mod, storage):
    response = papi.insert(project=project, body={'storageDataLocation': storage, \
                                                         'id':mod}).execute()

create_model(project = project_id , mod=model_name , storage=storage_data_location)


