from flask import Flask

from gevent.pywsgi import WSGIServer
import grpc
import numpy as np
import requests
import tensorflow as tf
import os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

@app.route('/')
def main():

	# Download image 
	img = image.load_img("cabbage.jpg", target_size=(150,150))
	# map to tensor array
	img_tensor = image.img_to_array(img)
	# Assign dimensions
	img_tensor = np.expand_dims(img_tensor, axis=0)
	# pass into workable variable
	data = img_tensor
	# mapped image
	#print(data)  
	  
	# establish channel to docker image container
	channel = grpc.insecure_channel('192.168.99.100:8500')
  	grpc.channel_ready_future(channel).result()
	# create variable for service that sends object to channel
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	# assign values to props of request
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'testmodel'
	request.model_spec.signature_name = 'serving_default'
	request.inputs['input_image'].CopyFrom(
	tf.make_tensor_proto(data,shape=[1,150,150,3])
	)
	#send request to docker image container
	result = stub.Predict(request,10.0)
	# response from model
	floats = np.array(list(result.outputs['dense_1/Softmax:0'].float_val)) 
  	max_ = floats.argmax()
  	print("\n")
  	print(floats)
  	print("\n")
  	print(max_)
  	print("\n")
  	return max_

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__=='__main__':
	#app.run()
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
