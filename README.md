# Gender Classifier from Name

A machine learning application composed of a LMST Keras model and a Flask API that predicts the gender of given first name. The endpoints have ability to retrain and serve the Machine Learning model. 

## How to run
**Locally**
1. Clone the TensorFlownNameClassifier repository.
2. Install the required modules and packages.
```
pip3 install -r requirements.txt
```
3. A pre-trained model should already be available. If it is not available, run the following command within the Gender_Classifier directory. 
```
python3 gender_classifier.py
```
4. Bootup the Flask API with 
```
python3 app.py
```
4. Navigate to http://0.0.0.0:5000 to see the swagger UI

**Docker**
1. In command prompt, build the Docker Image by first navigating to the installed directory with the Dockerfile and running:
```
 docker build -t gender_classifier 
```
2. Run the image by running the following command
```
 docker run -d -p 5000:5000  gender_classifier
 ```
3. Navigate to http://localhost:5000 to see the swagger UI

**How to hit the API Manually**
* If you are not using the Swagger UI, you can also hit the API by running the following command:
```
curl http://localhost:5000/api/classifyGender -d 'Name=<Choose a Name>'
```
* You can also do multiple requests by appending additional arguments:
```
curl http://localhost:5000/api/classifyGender -d 'Name=Ben' -d 'Name=Tim'
```

## Specifications and Methodologies

#### Tools used
* Python
* TensorFlow, Keras
* Flask API
* Docker

#### Dataset
* The data used to train the model can be found in the following link: [name_gender.csv](https://raw.githubusercontent.com/Raymond-Su/TensorFlowNameRecogniser/master/Gender_Classifier/name_gender.csv) 
* The dataset was composed of over 95,000 entries split into 60,304 female entries and 34,724 male entries. 
* The dataset was split into training (60%), validation (20%) and testing (20%) data. 

#### Pre-processing the data
It was important to convert any natural language input to a machine-readable vector format. A fixed input length was chosen to improve performance during training and achieve more stable weights. 
* First each name was converted into a vector using a process known as [One-Hot Encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f). Each word is represented by a vector of n binary subjects, where n is the number of different chars in the alphabet. 
* The length of the name vector was chosen to be the length of the longest name in the database. In our case this was 15. 

#### Model
[LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Long-short term memory) was used to predict the gender from a given first name. This RNN (Recurrent Neural Network) addresses the issue of reasoning about previous events to inform later ones - allowing information to persist. 

The model was built using *Sequential* Keras Model which allows you to stack multiple layers on top of each other. The number of layers, nodes and neurons was chosen using the [K-fold cross validation framework](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#K-fold_cross-validation)

* The first layer was a LTSM model connected to 230 hidden nodes. 
* The second layer was a Dropout layer, which prevents overfitting the model by ignoring randomly selected neurons during training. Hence, it reduces the sensitivity to the specific weights of individual neurons. 20% was selected as a compromise value to achieve acceptable model accuracy and prevent overfitting. 
* Multiple LTSM and Dropout layers were used to detect complex features. While this increases the accuracy of the model it also increased the complexity of training it. 
* A dense layer was used to reduce the shape of the prediction to match the desired output. In our case, we have two output labels (M and F). 
* The final layer was the [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) activation layer which was chosen to output a probability. 
* I chose to use the [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) (Adaptive Moment Estimation) optimizer as it helped me converge a bit faster than using pure SGD or RMSProp and didn't overfit my data. It also works well with little changes in the hyperparameters. 

An accuracy of **91.0%** was achieved. When observing the validation tests, some false predictions seemed to occur when people combined their first and middle names together within the name field. A good step would be clean the original dataset from these cases. 

## Acknowledgments
Sam Gleeson 


