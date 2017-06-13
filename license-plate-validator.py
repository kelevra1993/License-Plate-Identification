import tensorflow as tf
import sys  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import numpy as np
import functions as fun
import time

#CODE THAT SPEEDS UP TRAINING BY 37.5 PERCENT
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

####################################################
#FIRST WE DEFINE SOME HYPERPARAMETERS FOR OUR MODEL#
####################################################
#SESSION DEFINITION
sess = tf.InteractiveSession()
#Weight storage path 
model_path="D:/LP weights/weights.ckpt"
#Number of ctc_inputs and number of classes 
# The number is equal to 26 letters + 10 digits + blank label
num_classes=38
#Input height, here the height of the images is actually 50 
#images were depth "unconcanated" which yields 50 * 3 
input_feature_length=150
#Size of Input Batch
input_batch_size=10
#Number of LSTM BLOCK, be careful tensorflow loosely uses the term LSTMCELL to define an LSTMBLOCK !!! 
#Litterature diffenciates these two concepts
num_hidden_units=600
#Data file path
tfrecords_file="D:/LP data/valid2000.tfrecords"
#Number of iterations that we will run to process the whole tfrecord file, here we have an input batch of 10
#so running 200 iterations lets us process the 2000 images that are in the tfrecord file 
num_iterations=200
#This will check how many files are in the checkpoint that will be used for model evaluation
ckpt=tf.train.get_checkpoint_state("D:\LP weights")
models=ckpt.all_model_checkpoint_paths[:]
num_models=len(models)
print("We are evaluating : %d models" %(num_models))
#File that contains information of model perfomances that have been evaluated
#We will be filling this file iteratively
evaluation="license_plate_models.txt"

#################
#BATCH RETRIEVAL#
#################
with tf.name_scope('Input-producer'): 
	#QUEUE CREATION
	filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=None)

	#GETTING INPUT BATCHES
	pre_train_data,pre_label_data,pre_length_data=fun.read_and_decode(filename_queue,input_batch_size)

	#DATA INPUT PLACEHOLDERS 
	#We are going to be using a time_major==False for the tf.nn.biirectional_dynamic_rnn, so we will initially be working with BatchxTimeStepsxfeatures
	#Feature value is equal to 150 (3 color channels) for our input data and 9 for our license plate label
	#this will change in the future for we might use padding along the vertical axis to have the same number of 
	#features along this same axis, right now our data is already correctly segmented
	  
	input_images=tf.placeholder(tf.float32,shape=[None,None,input_feature_length],name='Train_input')
	train_length_data=tf.placeholder(tf.int32)
	sparse_indices=tf.placeholder(tf.int64)
	sparse_values=tf.placeholder(tf.int32)
	sparse_shape=tf.placeholder(tf.int64)
	sparse_label=tf.SparseTensor(sparse_indices,sparse_values,sparse_shape)
	dense_label=tf.sparse_tensor_to_dense(sparse_label)

#########################################
#Definition of weight matrix generators #
#########################################

def weight_variables(shape,identifier):
    initial=tf.truncated_normal(shape,dtype=tf.float32,stddev=0.1)
    return tf.Variable(initial,name=identifier)
    
def bias_variables(shape,identifier):
    initial=tf.constant(1.0,shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=True, name=identifier)


#########################################
#DEFINITION BIDIRECTIONAL NEURAL NETWORK#
#########################################

with tf.name_scope('Bidirectional-Layer'):
	#FORWARD AND BACKWARD PASS DEFINITION
	
	#Faster LSTM cells
	foward_pass_cells=[tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(4)]
	backward_pass_cells=[tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(4)]
	
	#Stacking of LSTM Layers
	outputs, output_state_fw, output_state_bw=tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
	cells_fw=foward_pass_cells,
	cells_bw=backward_pass_cells,
	inputs=input_images,
	sequence_length=train_length_data,
	dtype=tf.float32)

	#HERE WE ADD OUR FORWARD AND OUR BACKWARD PASS, 
	outputs=tf.split(outputs,2,axis=2)
	outputs=tf.reduce_sum(outputs, axis=0)
	outputs=tf.reshape(outputs, [-1, num_hidden_units])

with tf.name_scope('BIDIRECTIONAL-TO-CTC-LAYER'):
	#DEFINITION OF A WEIGHT AND BIAS VECTOR THAT WILL APPLY COMPUTATION ON BIDIRECTIONAL NETWORK OUTPUTS
	#PURPOSE BEING TO FEED THEM TO THE OUR CTC SOFTMAX LAYER
	LSTM_CTC_WEIGHTS=weight_variables([num_hidden_units,num_classes],'LSTM_CTC_WEIGHTS')
	LSTM_CTC_BIAS=bias_variables([num_classes],'LSTM_CTC_BIAS')

	#COMPUTATION AND RESHAPE
	inputs_ctc=(tf.matmul(outputs,LSTM_CTC_WEIGHTS))+LSTM_CTC_BIAS
	inputs_ctc=(tf.reshape(inputs_ctc,[input_batch_size,-1,num_classes]))

with tf.name_scope('CTC-SOFT-MAX-LAYER'):
	#TRANSPOSAL OF OUR CTC INPUTS TO MAKE THEM TIME MAJOR FOR OUR DECODER
	inputs_ctc=tf.transpose(inputs_ctc,(1,0,2))

with tf.name_scope('DECODER'):
	#COMPUTATION OF GREEDY DECODING AND BEAM SEARCH DECODING 
	#WE HAVE NOT YET DECIDED ON WHICH ONE OF THE TWO DECODERS WE WILL USE FOR FINAL TRAINING 
	decoded,log_probability=tf.nn.ctc_greedy_decoder(inputs_ctc,sequence_length=train_length_data,merge_repeated=True)
	decoded_beam,log_probability_beam=tf.nn.ctc_beam_search_decoder(inputs_ctc,sequence_length=train_length_data,beam_width=200, top_paths=2,merge_repeated=False)

	#TRANSFORMATION OF SPARSE OUTPUTS FROM OUR DECODERS
	decoded_dense=tf.sparse_tensor_to_dense(decoded[0])
	decoded_beam_dense=tf.sparse_tensor_to_dense(decoded_beam[0])

sparse_label=tf.cast(sparse_label,dtype=tf.int64)

#MODEL EVALUATION 
#For performance evaluation, every license plate that has an edit distance higher than one will be considered as a false identification 
acc=tf.edit_distance(decoded[0],sparse_label,normalize=False)
# acc=tf.reduce_mean(acc)

acc_beam=tf.edit_distance(decoded_beam[0],sparse_label,normalize=False)
# acc_beam=tf.reduce_mean(acc_beam)


	
#INITIALIZATION OF OUR VARIABLES
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#MODEL SAVING DEFINITION + FILE LIMIT
saver=tf.train.Saver()

print("\n"*2)
print("----------Tensorflow has been set----------")
print("\n"*2)
    
#QUEUE COORDINATORS FOR BATCH FETCHING 
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)

#MODEL TESTING
error_dump=False
models=["./weights/weights.ckpt"]
for model in models : 

	#File that contains model evaluation perfomance 
	target=open(evaluation,"a")
	
	#so we have a certain number of models that we are going to be testing.
	#First we check if there is a model, if so, we restore it
	print(model)
	if(os.path.isfile(model+".meta")):
		print("")
		print( "We found a previous model")
		print("Model weights are being restored.....")
		saver.restore(sess,model)
		print("Model weights have been restored")
		print("\n")
	else:
		print("")
		print("No model weights were found....")
		print("")
		print("We generate a new model")
		print("\n")
	#initializing accuracy for every model evaluation 
	final_beam_accuracy=0
	final_greedy_accuracy=0
	beam_error=0
	greedy_error=0
	##########################################
	#RUNNING THE BIDIRECTIONAL NEURAL NETWORK#
	##########################################

	#MIGHT WANT TO EVALUATE RUN TIME AFTER WE HAVE COMPLETELY ESTABLISHED THE MODEL 
	start=time.time()
	for i in range(num_iterations):
		state=100*(i+1)/num_iterations
		if(state%25==0):
			print("Evaluation of model %s is at %d%% "%(model,state))
		#FIRST RUN OUR BATCH FETCHING METHOD FOR PREPROCESSING (CASTING, TURNING DENSE VECTORS INTO SPARSE VECTORS)
		
		input_batch_data, input_label_data, length_batch_data=sess.run([pre_train_data,pre_label_data,pre_length_data])

		sparse_parameters=fun._create_sparse_tensor(input_label_data[:,0,:])

		
		#FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR VALIDATION EVALUATION DURING TRAINING 
		#OPTIMIZATION STEP 
		indices=sparse_parameters[0]
		values=sparse_parameters[1]
		dense_shape=sparse_parameters[2]
		
		accuracy_beam,accuracy_greedy,decoded_beam,decoded_greedy=sess.run(
		[acc_beam,acc,decoded_beam_dense,decoded_dense]
		,feed_dict={input_images : input_batch_data, train_length_data : length_batch_data,sparse_indices : indices ,sparse_values : values ,sparse_shape : dense_shape})
		
				
		#evaluation of beam decoding accuracy 
		for ind,a_b in enumerate(accuracy_beam):
			if( a_b!=0):
				beam_error=beam_error+1
				if(error_dump):
					cv2.imwrite("./errors/original_"+fun.recon_label(input_label_data[ind])+"_predicted_"+fun.recon_label(np.matrix(decoded_beam[ind]))+".tiff",fun.recon_image(input_batch_data[ind],length_batch_data[ind]))
		
		#evaluation of greedy decoding accuracy
		for ind,a_b in enumerate(accuracy_greedy):
			if( a_b!=0):
				greedy_error=greedy_error+1
				
	end=time.time()
	final_beam_accuracy=(100*(1.0-(beam_error/(input_batch_size * num_iterations ))))
	final_greedy_accuracy=(100*(1.0-(greedy_error/(input_batch_size * num_iterations ))))
	print("\n")
	print("-------------------------------------------------------------")
	print("we have %d beam errors"%(beam_error))
	print("We have %d greedy errors"%(greedy_error))
	print("Beam decoder yielded an accuracy of %.4f%%"%(final_beam_accuracy))
	print("Greedy decoder yielded an accuracy of %.4f%%"%(final_greedy_accuracy))
	print("The model was shown %d images"%(input_batch_size * num_iterations))
	#Dumping information into our evaluation file
	# target.write("Evaluation of model %s yields a beam decoding accuracy of %.4f%%"%(model,final_beam_accuracy))
	# target.write("\n")
	# target.write("Evaluation of model %s yields a beam decoding accuracy of %.4f%%"%(model,final_greedy_accuracy))
	# target.write("\n"*2)
	# target.close()
	print("these %d iterations took %d seconds"%(num_iterations,(end-start)))
	start=time.time()
	print("-------------------------------------------------------------")
	print("\n"*2)

		