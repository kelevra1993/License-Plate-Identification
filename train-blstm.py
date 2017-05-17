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
num_iterations=100000

#################
#BATCH RETRIEVAL#
#################

#QUEUE CREATION
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=None,shuffle=True)

#WILL BE FURTHER REMOVED FROM THE FILE WHEN WE ARE DONE WITH THE PLACEHOLDERS
#GETTING INPUT BATCHES
train_data,label_data,length_data=fun.read_and_decode(filename_queue,input_batch_size)

#DATA INPUT PLACEHOLDERS 
#We are going to be using a time_major==False for the tf.nn.biirectional_dynamic_rnn, so we will initially be working with BatchxTimeStepsxfeatures
#Feature value is equal to 150 (3 color channels) for our input data and 9 for our license plate label
  
# input_images=tf.placeholder(tf.float32,shape=[None,input_feature_length,None],name='Train_input')
input_images=train_data
sparse_label=label_data
dense_label=tf.sparse_tensor_to_dense(sparse_label)
train_length_data=length_data
# train_length_data=tf.placeholder(tf.int64,shape=[None])
# train_length_data=tf.cast(train_length_data,dtype=tf.int32)
# sparse_label=tf.sparse_placeholder(tf.int32)
#########################################
#Definition of weight matrix generators #
#########################################

def weight_variables(shape,identifier):
    initial=tf.truncated_normal(shape,dtype=tf.float32,stddev=0.1)
    return tf.Variable(initial,name=identifier)
    
def bias_variables(shape,identifier):
    initial=tf.constant(1.0,shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=True, name=identifier)

    
#HAS TO BE REMOVED WHEN THE MODEL IS DONE BEING ESTABLISHED
# one=tf.contrib.learn.run_n({"a":train_data,"b":length_data},n=1,feed_dict=None)
# print(one[0]["a"].shape)
# print(one[0]["b"].shape)


#########################################
#DEFINITION BIDIRECTIONAL NEURAL NETWORK#
#########################################

#FORWARD AND BACKWARD PASS DEFINITION
foward_pass=tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True)
backward_pass=tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True)

#DEFINING FOWARD OUTPUTS AND BACKWARD OUTPUTS 
#DEFING FORWARD AND BACKWARD HIDDEN AND CELL STATES
outputs, states=tf.nn.bidirectional_dynamic_rnn(
cell_fw=foward_pass,
cell_bw=backward_pass,
inputs=input_images,
sequence_length=train_length_data,
dtype=tf.float32)


#HERE WE ADD OUR FORWARD AND OUR BACKWARD PASS, 
# WE MIGHT HAVE TO CHANGE THIS METHOD AFTERWARDS MAYBE USE SOME KIND OF CONCATENATION
#First we reshape our outputs before doing a fully connected feedfoward on them
#Initially outputs is a tuple, we chose to concatenate foward and backward outputs 
# (foward_outputs, backward_outputs)=outputs
outputs=tf.reduce_sum(outputs, axis=0)
outputs=tf.reshape(outputs, [-1, num_hidden_units])


#DEFINITION OF A WEIGHT AND BIAS VECTOR THAT WILL APPLY COMPUTATION ON BIDIRECTIONAL NETWORK OUTPUTS
#PURPOSE BEING TO FEED THEM TO THE OUR CTC SOFTMAX LAYER
LSTM_CTC_WEIGHTS=weight_variables([num_hidden_units,num_classes],'weights')
LSTM_CTC_BIAS=bias_variables([num_classes],'bias')

#COMPUTATION AND RESHAPE
inputs_ctc=(tf.matmul(outputs,LSTM_CTC_WEIGHTS))+LSTM_CTC_BIAS
inputs_ctc=(tf.reshape(inputs_ctc,[input_batch_size,-1,num_classes]))


#LINE WILL BE DELT WITH IN THE ITERATION PROCESS THAT HAS TO DO WITH THE SPARSING OF TENSORS 
# label=tf.contrib.learn.run_n({"a":label_data[0]},n=1,feed_dict=None)
# print("\n"*5)
# print(fun.recon_label(label[0]["a"]))
# sparse_label=fun._create_sparse_tensor(label[0]["a"])

#CTC LOSS COMPUTATION 
cost=tf.nn.ctc_loss(inputs=inputs_ctc, labels=sparse_label, sequence_length=train_length_data, preprocess_collapse_repeated=False, ctc_merge_repeated=True,time_major=False)

#TRANSPOSAL OF OUR CTC INPUTS TO MAKE THEM TIME MAJOR FOR OUR DECODER
inputs_ctc=tf.transpose(inputs_ctc,(1,0,2))

#COMPUTATION OF GREEDY DECODING AND BEAM SEARCH DECODING 
#WE HAVE NOT YET DECIDED ON WHICH ONE OF THE TWO DECODERS WE WILL USE FOR FINAL TRAINING 
decoded,log_probability=tf.nn.ctc_greedy_decoder(inputs_ctc,sequence_length=train_length_data,merge_repeated=False)
decoded_beam,log_probability_beam=tf.nn.ctc_beam_search_decoder(inputs_ctc,sequence_length=train_length_data,beam_width=200, top_paths=2,merge_repeated=False)

#TRANSFORMATION OF SPARSE OUTPUTS FROM OUR DECODERS
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

print("\n"*30)
print("----------Tensorflow has been set----------")
print("\n"*10)

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

	# input_batch_data, input_label_data, length_batch_data=sess.run([train_data,label_data,length_data])
    #FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR VALIDATION EVALUATION DURING TRAINING 
	#OPTIMIZATION STEP 
	
	len,_,loss,accuracy_beam,accuracy_greedy,decoded_beam,decoded_greedy,lab=sess.run([train_length_data,train_step,cost,acc_beam,acc,decoded_beam_dense,decoded_dense,dense_label],feed_dict={})

	print("-------------------------------------------------------------")
	print("we called the model %d times"%(i+1))
	print("The current loss is : ",loss)
	print("The current len is : ",len[0])
	# print("The accuracy from the greedy decoding is : %d%%"%(accuracy_greedy))
	# print("The network was shown a license plate : ",fun.recon_label(input_label_data[0]))
	# print("The network predicted a license plate : ",fun.recon_label(decoded_greedy))
	# print("\n"*2)
	print("The edit distance is : %d"%(accuracy_beam))
	print("The network was shown a license plate : ",fun.recon_label(lab))
	print("The network predicted a license plate : ",fun.recon_label(decoded_beam))
	print("-------------------------------------------------------------")
	print("\n"*2)










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


# results=tf.contrib.learn.run_n({"a":foward_outputs,
								# "b":backward_outputs,
								# "c":foward_output_state,
								# "d":backward_output_state,
								# "e":outputs,
								# "f":inputs_ctc,
								# "cost":cost,
								# "greed":decoded_dense,
								# "beam":decoded_beam_dense,
								# "log_greed":log_probability,
								# "log_beam":log_probability_beam,
								# "acc":acc,
								# "acc_beam":acc_beam   ,
								# },
								# n=1,feed_dict=None)

								
								
# print(results[0]["a"].shape)								
# print(results[0]["b"].shape)								
# print(results[0]["c"].h.shape)								
# print(results[0]["c"].c.shape)								
# print(results[0]["d"].h.shape)								
# print(results[0]["d"].c.shape)
# print(results[0]["e"].shape)
# print("\n"*5)
# print(results[0]["cost"])
# print(results[0]["cost"].shape)
# print(results[0]["greed"])
# print("GREEDY DECODER GIVES OUT :" ,fun.recon_label(results[0]["greed"]))
# print(results[0]["beam"])
# print("BEAM SEARCH DECODER GIVES OUT :",fun.recon_label(results[0]["beam"]))
# print(results[0]["log_greed"])
# print(results[0]["log_beam"])
# print(results[0]["acc"])
# print(results[0]["acc_beam"])

# print(results[0]["a"])									
# print(results[0]["b"])							
# outputs=(foward_outputs, backward_outputs)

