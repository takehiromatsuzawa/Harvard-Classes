from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import sys
import json

# python get_status.py model_name project_id credential_file_name
# Example: python get_status.py job-identifier-1-2 'data-science-153816' Data-Science-6af3c695f128.json 

scopes = ['https://www.googleapis.com/auth/prediction', 'https://www.googleapis.com/auth/devstorage.full_control', \
          'https://www.googleapis.com/auth/cloud-platform']

# @model_name: model created in create_model.py
model_name           = sys.argv[1]

# @project_id: reference: https://support.google.com/cloud/answer/6158840?hl=en
project_id         = sys.argv[2]

# @credential_file_name: reference: http://gspread.readthedocs.io/en/latest/oauth2.html
credential_file_name = sys.argv[3]

credentials = ServiceAccountCredentials.from_json_keyfile_name(credential_file_name, scopes=scopes)
http_auth = credentials.authorize(Http())
service = build('prediction', 'v1.6', http = http_auth)
papi = service.trainedmodels() 

def get_model(project, mod):
    response = papi.get(project = project, id = mod).execute()
    print json.dumps(response, sort_keys=True, indent=4)
    
get_model(project = project_id, mod = model_name)

