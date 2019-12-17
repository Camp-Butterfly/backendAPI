from flask import Flask
from flask import request

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

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

@app.route('/api/v1/model', methods=['POST'])
def image_post():
	#get base-64 from json object
	test = request.get_json(force=True)
	print(test)
	img_c = test['image_content']

	#preprocessing for base64 encoded image which has to do what 
	#image.load_img does => opens file, resizes to target size then maps to a keras array
	###
	img_c = base64.b64decode(img_c)
	buf = io.BytesIO(img_c)
	img = Image.open(buf)
	img = img.resize([150,150])
	###
	img_tensor = image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	data = img_tensor

	#instantiate request
	channel = grpc.insecure_channel('192.168.99.100:8500')
	grpc.channel_ready_future(channel).result()
	# create variable for service that sends object to channel
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	# assign values to props of request
	req = predict_pb2.PredictRequest()
	req.model_spec.name = 'testmodel'
	req.model_spec.signature_name = 'serving_default'
	req.inputs['input_image'].CopyFrom(
	tf.make_tensor_proto(data,shape=[1,150,150,3])
	)

	#make request to docker image container
	result = stub.Predict(req,10.0)
	
	#response from model as tensorflow array
	floats = np.array(list(result.outputs['dense_1/Softmax:0'].float_val)) 
  	#empty response catch
  	if(not floats.argmax()):
  		max_ = 4
  	else:
  		max_ = floats.argmax()

  	print("\n")
  	print(floats)
  	print("\n")
  	print(max_)
  	print("\n")
  	# convert numpy integer to json; response to React app
  	res = json.dumps(max_)
	return res

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

#if __name__=='__main__':
app.run()
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
