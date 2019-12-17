from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

from gevent.pywsgi import WSGIServer
import grpc
import numpy as np
import requests
import tensorflow as tf
import os
import base64
import io
import PIL
import json
import six

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.preprocessing import image
from PIL import Image

from make_tensor_proto import make_tensor_proto

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/v1/model', methods=['POST'])
@cross_origin()
def image_post():
	test = request.get_json(force=True)
	#print(test)
	img_c = test['image_content']
	img_c = base64.b64decode(img_c)
	buf = io.BytesIO(img_c)
	img = Image.open(buf)
	img = img.resize([150,150])
	img_tensor = image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	data = img_tensor

	channel = grpc.insecure_channel('34.68.117.217:8500')
	# create variable for service that sends object to channel
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	# assign values to props of request
	req = predict_pb2.PredictRequest()
	req.model_spec.name = 'test2'
	req.model_spec.signature_name = 'serving_default'
	req.inputs['input_image'].CopyFrom(
	make_tensor_proto(data,shape=[1,150,150,3])
	)

	#send request to docker image container
	result = stub.Predict(req,10.0)
	# response from model
	floats = np.array(list(result.outputs['dense_1/Softmax:0'].float_val)) 
  	max_ = floats.argmax()
  	#print(floats)
  	#print("\n")
  	#print(max_)
  	#print("\n")
  	res = json.dumps(max_)
	return res

@app.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"

#@app.after_request
#def after_request(response):
#  response.headers.add('Access-Control-Allow-Origin', '*')
#  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#  return response

#if __name__=='__main__':
#app.run()
#http_server = WSGIServer(('', 5000), app)
#http_server.serve_forever()
