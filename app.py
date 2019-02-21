from flask import Flask,jsonify
from flask_restplus import Api, fields, Resource
from pathlib import Path
from Gender_Classifier.gender_classifier import normalize, name_encoding
import tensorflow as tf
import numpy as np

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Gender Classifer API',
    description='An API that classifies the gender given a first name')

ns = api.namespace('ClassifyGender')

class NullableString(fields.String):
    __schema_type__ = ['string', 'null']
    __schema_example__ = 'nullable string'
    
parser = api.parser()
parser.add_argument(
    'Name', 
    required=True, 
    help='The persons name',
    location='form',
    action='append')

@ns.route('/')
class GenderApi(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args(strict=True)
        nameList = args["Name"]
        resultList = []
        for name in nameList:
            if(name.isalpha()):
                result = self.get_result(name)
                resultList.append(result)
            else:
                return app.response_class(response="Error",status=404)
        
        response = jsonify(resultList)
        response.status_code=200
        return response

    def get_result(self, name):
        model_dir = Path("Gender_Classifier/gender_model.h5")
        model = tf.keras.models.load_model(model_dir)
        result = {'Name':name,'Male':0, 'Female':0}
        nameList = []
        nameList.append(name)
        prediction = model.predict(np.asarray([np.asarray(name_encoding(normalize(name))) for name in nameList]))
        result['Male'] = (prediction.tolist())[0][0]
        result['Female'] = (prediction.tolist())[0][1]
        return result

if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)