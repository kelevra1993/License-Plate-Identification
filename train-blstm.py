import tensorflow as tf
import sys 
import os
import cv2
import numpy as np
import functions as fun



#CODE THAT SPEEDS UP TRAINING BY 37.5 PERCENT
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


####################################################
#FIRST WE DEFINE SOME HYPERPARAMETERS FOR OUR MODEL#
####################################################
#SESSION DEFINITION
sess = tf.InteractiveSession()

#Network parameters
learning_rate=1e-4
momentum=0.9
#Number of ctc_inputs and number of classes
num_classes=38
#Input height and width
input_feature_length=150
#Size of Input Batch
input_batch_size=1
#Number of LSTM BLOCK, be careful tensorflow loosely uses the term LSTMCELL to define an LSTMBLOCK !!! 
#Litterature diffentiates these two concepts
num_hidden_units=200
#Data file
tfrecords_file="./data/train.tfrecords"
#Number of iterations that we will run 
num_iterations=20

#Definition of input images 
#we might need to use grayscale
#We are going to be using a time_major==False for the tf.nn.biirectional_dynamic_rnn, so we will initially be working with BatchxTimeStepsxfeatures
#Feature value is equal to 150 (3 color channels) for our input data and 9 for our license plate label
  
# train_data=tf.placeholder(tf.int64,shape=[None,input_feature_length,None],name='Train_input')
# train_data_length=tf.placeholder(tf.int64,shape=[None])

#########################################
#Definition of weight matrix generators #
#########################################

def weight_variables(shape,identifier):
    initial=tf.truncated_normal(shape,dtype=tf.float32,stddev=0.1)
    return tf.Variable(initial,name=identifier)
    
def bias_variables(shape,identifier):
    initial=tf.constant(1.0,shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=True, name=identifier)


#################
#Network TESTING#
#################
#Creation of a queue, working with 10 epochs so 10*100 images, an image will basically be shown 10 times
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=None)
#Get an image batch
train_data,label_batch,train_data_length=fun.read_and_decode(filename_queue,input_batch_size)
train_data=tf.cast(train_data,dtype=tf.float32)
label_batch=tf.cast(label_batch,dtype=tf.int32)
train_data_length=tf.cast(train_data_length,dtype=tf.int32)

one=tf.contrib.learn.run_n({"a":train_data,"b":train_data_length},n=1,feed_dict=None)
print(one[0]["a"].shape)
print(one[0]["b"].shape)

################################
#Definition of the LSTM Network#
################################

#Definition of foward and backward pass
foward_pass=tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True)
backward_pass=tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True)

#Computation of output when input goes through a foward and backward pass in the Bidirectionnal network
outputs, states=tf.nn.bidirectional_dynamic_rnn(
cell_fw=foward_pass,
cell_bw=backward_pass,
inputs=train_data,
sequence_length=train_data_length,
dtype=tf.float32)
(foward_outputs, backward_outputs)=outputs
(foward_output_state,backward_output_state)=states

#First we reshape our outputs before doing a fully connected feedfoward on them
#Initially outputs is a tuple, we chose to concatenate foward and backward outputs 
outputs=tf.reduce_sum(outputs, axis=0)
outputs=tf.reshape(outputs, [-1, num_hidden_units])

#Creation of an weight vector that will further feed input to the CTC layer
LSTM_CTC_WEIGHTS=weight_variables([num_hidden_units,num_classes],'weights')
LSTM_CTC_BIAS=bias_variables([num_classes],'bias')

inputs_ctc=(tf.matmul(outputs,LSTM_CTC_WEIGHTS))+LSTM_CTC_BIAS
inputs_ctc=(tf.reshape(inputs_ctc,[input_batch_size,-1,num_classes]))

#Computation of CTC loss

label=tf.contrib.learn.run_n({"a":label_batch[0]},n=1,feed_dict=None)
print("\n"*5)
print(fun.recon_label(label[0]["a"]))
sparse_label=fun._create_sparse_tensor(label[0]["a"])
# print("\n"*5)
# inputs_ctc=np.zeros((1,265,38))
# labelling=[[
# 13,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 13,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 36,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 0,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 1,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 1,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 36,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 13,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,
# 37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,13
# ]]
# labelling=[[13,13,35,0,1,1,36,13,13]]
# print(len(labelling[0]))
# j=0
# for k in labelling[0]:
    # inputs_ctc[0,j,k]=100
    # j=j+1
# print(inputs_ctc)
# print(inputs_ctc.shape)
# inputs_ctc=tf.cast(tf.stack(inputs_ctc),dtype=tf.float32)

cost=tf.nn.ctc_loss(inputs=inputs_ctc, labels=sparse_label, sequence_length=train_data_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True,time_major=False)

inputs_ctc=tf.transpose(inputs_ctc,(1,0,2))

decoded,log_probability=tf.nn.ctc_greedy_decoder(inputs_ctc,sequence_length=train_data_length,merge_repeated=False)
decoded_beam,log_probability_beam=tf.nn.ctc_beam_search_decoder(inputs_ctc,sequence_length=train_data_length,beam_width=200, top_paths=2,merge_repeated=False)

decoded_dense=tf.sparse_tensor_to_dense(decoded[0])
decoded_beam_dense=tf.sparse_tensor_to_dense(decoded_beam[0])

sparse_label=tf.cast(sparse_label,dtype=tf.int64)


#MODEL EVALUATION 
acc=tf.edit_distance(decoded[0],sparse_label)
acc_beam=tf.edit_distance(decoded_beam[0],sparse_label,normalize=False)
train_step=tf.train.AdamOptimizer(learning_rate,momentum).minimize(cost)


#INITIALIZATION OF OUR VARIABLES
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


#QUEUE COORDINATORS FOR BATCH FETCHING 
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)


##########################################
#RUNNING THE BIDIRECTIONAL NEURAL NETWORK#
##########################################

#MIGHT WANT TO EVALUATE RUN TIME AFTER WE HAVE COMPLETELY ESTABLISHED THE MODEL 

for i in range(num_iterations):
    
    #FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR MODEL SAVING PATHS
    
    #FIRST RUN OUR BATCH FETCHING METHOD FOR PREPROCESSING (CASTING, TURNING DENSE VECTORS INTO SPARSE VECTORS)
    
    input_data_batch,input_label_batch=
    
    #FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR VALIDATION EVALUATION DURING TRAINING 




















































results=tf.contrib.learn.run_n({"a":foward_outputs,
								"b":backward_outputs,
								"c":foward_output_state,
								"d":backward_output_state,
								"e":outputs,
								"f":inputs_ctc,
								"cost":cost,
								"greed":decoded_dense,
								"beam":decoded_beam_dense,
								"log_greed":log_probability,
								"log_beam":log_probability_beam,
								"acc":acc,
								"acc_beam":acc_beam   ,
								},
								n=1,feed_dict=None)

								
								
# print(results[0]["a"].shape)								
# print(results[0]["b"].shape)								
# print(results[0]["c"].h.shape)								
# print(results[0]["c"].c.shape)								
# print(results[0]["d"].h.shape)								
# print(results[0]["d"].c.shape)
# print(results[0]["e"].shape)
print("\n"*5)
print(results[0]["cost"])
print(results[0]["cost"].shape)
print(results[0]["greed"])
print("GREEDY DECODER GIVES OUT :" ,fun.recon_label(results[0]["greed"]))
print(results[0]["beam"])
print("BEAM SEARCH DECODER GIVES OUT :",fun.recon_label(results[0]["beam"]))
print(results[0]["log_greed"])
print(results[0]["log_beam"])
print(results[0]["acc"])
print(results[0]["acc_beam"])

# print(results[0]["a"])									
# print(results[0]["b"])							
# outputs=(foward_outputs, backward_outputs)

