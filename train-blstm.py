import tensorflow as tf
import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import numpy as np
import functions as fun
import time
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#CODE THAT SPEEDS UP TRAINING BY 37.5 PERCENT
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

####################################################
#FIRST WE DEFINE SOME HYPERPARAMETERS FOR OUR MODEL#
####################################################
#SESSION DEFINITION
sess = tf.InteractiveSession()

#Weight storage
model_path="./weights/weights"
weight_saver=5000

#Tensorboard information
tensorboard_path="./tensorboard"

#Network parameters
learning_rate=1e-4
momentum=0.9

#Number of ctc_inputs and number of classes
num_classes=38

#Input height
input_feature_length=150

#Size of Input Batch
input_batch_size=1

#Number of LSTM BLOCK, be careful tensorflow loosely uses the term LSTMCELL to define an LSTMBLOCK !!! 
#Litterature diffenciates these two concepts
num_hidden_units=200

#Data files....
'''WE WILL NEED ONE FOR A VALIDATION SET WILL COME LATER IN THE FUTURE'''
tfrecords_file="./data/train.tfrecords"

#Number of iterations that we will run 
num_iterations=5000000

#Dropout for validation
keep_probability=0.5

#Information Dump to our terminal
info_dump=1000

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
	label_placeholder=tf.placeholder(tf.float32,shape=[None,None,num_classes])
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
    initial=tf.truncated_normal(shape,dtype=tf.float32,stddev=0.01)
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
	#foward_pass_cells=[tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(2)]
	#backward_pass_cells=[tf.contrib.rnn.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(2)]

	foward_pass_cells=[tf.nn.rnn_cell.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(2)]
	backward_pass_cells=[tf.nn.rnn_cell.LSTMCell(num_units=num_hidden_units,use_peepholes=True) for i in range(2)]

	#Stacking of LSTM Layers
	outputs, output_state_fw, output_state_bw=tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
	cells_fw=foward_pass_cells,
	cells_bw=backward_pass_cells,
	inputs=input_images,
	sequence_length=train_length_data,
	parallel_iterations=2,
	dtype=tf.float32)

	#HERE WE ADD OUR FORWARD AND OUR BACKWARD PASS,
	evaluator=outputs
	outputs=tf.split(outputs,2,axis=2)
	outputs=tf.reduce_sum(outputs, axis=0)
	after_reduce=outputs
	outputs=tf.reshape(outputs, [-1, num_hidden_units])

with tf.name_scope('BIDIRECTIONAL-TO-CTC-LAYER'):
    #DEFINITION OF A WEIGHT AND BIAS VECTOR THAT WILL APPLY COMPUTATION ON BIDIRECTIONAL NETWORK OUTPUTS
    #PURPOSE BEING TO FEED THEM TO THE OUR CTC SOFTMAX LAYER
    LSTM_CTC_WEIGHTS=weight_variables([num_hidden_units,num_classes],'LSTM_CTC_WEIGHTS')
    LSTM_CTC_BIAS=bias_variables([num_classes],'LSTM_CTC_BIAS')

    #COMPUTATION AND RESHAPE
    inputs_ctc=(tf.matmul(outputs,LSTM_CTC_WEIGHTS))+LSTM_CTC_BIAS
    inputs_ctc=(tf.reshape(inputs_ctc,[input_batch_size,-1,num_classes]))
    inp=inputs_ctc

    soft=tf.nn.softmax(inp)
    np.set_printoptions(linewidth=200)

    predictions=tf.argmax(inp,axis=2,name='predictions')
    lab=tf.argmax(label_placeholder,axis=2,name='label_indices')

    recall,recall_update_op=tf.metrics.recall(labels=lab[0],predictions=predictions[0])
    precision,precision_update_op=tf.metrics.precision(labels=lab[0],predictions=predictions[0])
    absolute=tf.contrib.metrics.streaming_mean_absolute_error(labels=lab[0],predictions=predictions[0])

    entropy=tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder,logits=inp)

with tf.name_scope('CTC-SOFT-MAX-LAYER'):
	#CTC LOSS COMPUTATION 
	#cost=tf.nn.ctc_loss(inputs=inputs_ctc, labels=sparse_label, sequence_length=train_length_data, preprocess_collapse_repeated=False, ctc_merge_repeated=True,time_major=False)
	#cost = tf.reduce_mean(cost)
	cost=tf.reduce_mean(entropy)
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
with tf.name_scope('Model_Evaluation'):
	sparse_label=tf.cast(sparse_label,dtype=tf.int64)

	#MODEL EVALUATION
	acc=tf.edit_distance(decoded[0],sparse_label,normalize=False)
	acc=tf.reduce_mean(acc)

	acc_beam=tf.edit_distance(decoded_beam[0],sparse_label,normalize=False)
	acc_beam=tf.reduce_mean(acc_beam)

with tf.name_scope('Gradient-computation'):
	train_step=tf.train.AdamOptimizer(learning_rate,momentum).minimize(cost)



############################
#TensordBoard Visualization#
############################


cost_view=tf.summary.scalar("cross_entropy",cost)

accuracy_view=tf.summary.scalar("accuracy_greedy",acc)
accuracy_view=tf.summary.scalar("accuracy_beam",acc_beam)

#merge all of the variables for visualization
merged=tf.summary.merge_all()

mixed_writer=tf.summary.FileWriter(tensorboard_path,sess.graph)
	
	
	
#INITIALIZATION OF OUR VARIABLES
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#MODEL SAVING DEFINITION + FILE LIMIT
saver=tf.train.Saver(max_to_keep=100000)
print("----------Tensorflow has been set----------")

#MODEL CHECK AND RESTORATION
if(os.path.isfile(model_path+".meta")):
    print("")
    print( "We found a previous model")
    print("Model weights are being restored.....")
    saver.restore(sess,model_path)
    print("Model weights have been restored")
    print("\n"*5)
else:
    print("")
    print("No model weights were found....")
    print("")
    print("We generate a new model")
    print("\n"*5)
    
#QUEUE COORDINATORS FOR BATCH FETCHING 
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)


##########################################
#RUNNING THE BIDIRECTIONAL NEURAL NETWORK#
##########################################

#MIGHT WANT TO EVALUATE RUN TIME AFTER WE HAVE COMPLETELY ESTABLISHED THE MODEL 
start=time.time()
for i in range(2):
    
    #FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR MODEL SAVING PATHS

    #FIRST RUN OUR BATCH FETCHING METHOD FOR PREPROCESSING (CASTING, TURNING DENSE VECTORS INTO SPARSE VECTORS)

    input_batch_data, input_label_data, length_batch_data=sess.run([pre_train_data,pre_label_data,pre_length_data])

    sparse_parameters=fun._create_sparse_tensor(input_label_data[:,0,:])
    l=length_batch_data[0]

    label = np.zeros((1,l , 38))
    for x in range(l):
        label[0, x, 12] = 1
    print(np.argmax(label,axis=2))

    #FURTHER DOWN THE PIPELINE WE WILL DEFINE OUR VALIDATION EVALUATION DURING TRAINING
    #OPTIMIZATION STEP
    indices=sparse_parameters[0]
    values=sparse_parameters[1]
    dense_shape=sparse_parameters[2]

    ab,rec,pre,pred,ent,so,ins,out,after,eval,_,loss,accuracy_beam,accuracy_greedy,decoded_beam,decoded_greedy,summary=sess.run(
    [absolute,recall,precision,predictions,entropy,soft,inp,outputs,after_reduce,evaluator,train_step,cost,acc_beam,acc,decoded_beam_dense,decoded_dense,merged]
    ,feed_dict={label_placeholder:label,input_images : input_batch_data, train_length_data : length_batch_data,sparse_indices : indices ,sparse_values : values ,sparse_shape : dense_shape})


    #print(ent.shape)
    #print(ent[0,0:10])
    #print("The minimum loss is :",loss)
    print(pred[0])
    #sys.exit()
    """if(i%10==0):
        print("The softmax per column gives out")
        print(so[0, 0, :])"""
    y_true=np.argmax(label,axis=2)
    y_pred=pred

    y_true = np.argmax(label, axis=2)[0]
    y_pred = pred[0]



    print("\n")
    print("TENSORFLOW RESULTS")

    one = recall_score(y_true, y_pred, average='macro')
    print(one)
    two = precision_score(y_true, y_pred, average='macro')
    print(two)
    print(2*(one*two)/(one+two))
    score = f1_score(y_true=y_true,y_pred=y_pred, average='macro')
    print("macro gives out",score)

    print("Recall is :",rec)
    print("Precision is :",pre)
    print("Absolute is :",ab)

    time.sleep(5)
    if((i+1)%info_dump==0):
        print("-------------------------------------------------------------")
        print("we called the model %d times"%(i+1))
        print("The current loss is : ",loss)
        print("The mean edit distance is for beam decoding is : ",(accuracy_beam))
        print("The mean edit distance is for greedy decoding is : ",(accuracy_greedy))
        decoded_beam=np.asarray(decoded_beam, dtype=np.int32)
        decoded_greedy=np.asarray(decoded_greedy, dtype=np.int32)

        for j in range(2):
            print("\n")
            print("The network was shown a license plate : ",fun.recon_label(input_label_data[j]))
            print("Beam decode predicted a license plate : ",fun.recon_label(np.matrix(decoded_beam[j])))
            print("Greedy decode predicted a license plate : ",fun.recon_label(np.matrix(decoded_greedy[j])))
        end=time.time()
        print("these %d iterations took %d seconds"%(info_dump,(end-start)))
        start=time.time()
        print("-------------------------------------------------------------")
        print("\n")
    mixed_writer.add_summary(summary,i)
    if((i+1)%weight_saver==0):
        print("we are at iteration %d so we are going to save the model"%(i+1))
        print("model is being saved.....")
        save_path=saver.save(sess,model_path+"_iteration_%d.ckpt"%(i+1))
        print("model has been saved succesfully")

