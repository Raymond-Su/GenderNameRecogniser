from flask import Flask
from flask_restplus import Api, fields, Resource
import tensorflow as tf
import numpy as np
import string

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Gender Classifer API',
    description='An API that classifies the gender given a first name')

ns = api.namespace('ClassifyGender')

parser = api.parser()
parser.add_argument(
    'Name', 
    type= str, 
    required=True, 
    help='The persons name',
    location='form')

@ns.route('/')
class GenderApi(Resource):
    @api.doc(parser=parser)
    
    def post(self):
        args = parser.parse_args()
        name = args["Name"]
        if(name.isalpha()):
            result = self.get_result(name)
            return result, 200
        else:
            response = app.response_class(response="Error",status=404)
            return response
        

    accepted_chars = 'abcdefghijklmnopqrstuvwxyz'

    def get_result(self, name):
        model = tf.keras.models.load_model('./gender_model.h5')
        confidence = {'Male':0, 'Female':0}
        nameList = []
        nameList.append(name)
        prediction = model.predict(np.asarray([np.asarray(self.name_encoding(self.normalize(name))) for name in nameList]))
        confidence['Male'] = (prediction.tolist())[0][0]
        confidence['Female'] = (prediction.tolist())[0][1]
        return {
            "Name": name,
            "Male": confidence['Male'],
            "Female": confidence['Female']
        }

    def normalize(self,line):
        return [c.lower() for c in line if c.lower() in self.accepted_chars]

    def name_encoding(self, name):
        max_len = 15
        char_to_int = dict((c, i) for i, c in enumerate(self.accepted_chars))
        int_to_char = dict((i, c) for i, c in enumerate(self.accepted_chars))
        # Encode input data to int, e.g. a->1, z->26
        integer_encoded = [char_to_int[char] for i, char in enumerate(name) if i < 26]
        onehot_encoded = list()
        
        for value in integer_encoded:
            # create a list of n zeros, where n is equal to the number of accepted characters
            letter = [0 for _ in range(len(self.accepted_chars))]
            letter[value] = 1
            onehot_encoded.append(letter)
            
        # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
        for _ in range(max_len - len(name)):
            onehot_encoded.append([0 for _ in range(len(self.accepted_chars))])
            
        return onehot_encoded

if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)