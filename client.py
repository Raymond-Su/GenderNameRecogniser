import json
import requests

api_url = 'http://0.0.0.0:5000/ClassifyGender/'
data = {'name':'Joel'}
r = requests.post(url=api_url, data=data)
print(r.status_code,r.text)