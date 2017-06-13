import cv2
import numpy as np
import tensorflow as tf
 
#we have 37 labels (digits,alphabets and "-") and one blank label for ctc 
num_classes=38
 
def recon_label(label):
	#Our final string plate from the hot vector
	full_label=[]
	#First get the number of characters in the plate
	label_length=label.shape[1]
	#Now get the indices of our characters
	for col in range(label_length):
		#Dealing with number characters
		ind=label[0,col]
		if(ind<10):
			character=chr(ind+48)
			full_label.append(character)
		#Dealing with characters
		if(ind>9 and ind<36):
			character=chr((ind+55))
			full_label.append(character)
		#Dealing with '-' character
		if(ind==36):
			character=chr((45))
			full_label.append(character)
	full_label=''.join(full_label)
	return(full_label)
	
def recon_image(image,column_size):
    image=np.transpose(image)
    buffer=np.zeros((50,column_size,3))
    for i in range(150):
        if(i<50):
            for j in range(column_size):
                buffer[i,j,0]=image[i,j]
        if(i>49 and i<100):
            for j in range(column_size):
                buffer[i-50,j,1]=image[i,j]
        if(i>99):
            for j in range(column_size):
                buffer[i-100,j,2]=image[i,j]
    # return(buffer/255)
    return(buffer)
	
	
#function used to create a sparse tensor for ctc
def _create_sparse_tensor(label_batch):

	idx=np.where( label_batch > -1 )
	# indices=np.stack((idx[0],idx[1],idx[2]),axis=-1)
	indices=np.stack((idx[0],idx[1]),axis=-1)
	label_batch=np.asarray(label_batch, dtype=np.int32)
	# values=np.reshape(label_batch,label_batch.shape[0]*label_batch.shape[1]*label_batch.shape[2])
	values=np.reshape(label_batch,label_batch.shape[0]*label_batch.shape[1])
	# dense_shape=(label_batch.shape[0],label_batch.shape[1],label_batch.shape[2])
	dense_shape=(label_batch.shape[0],label_batch.shape[1])
	return (indices,values,dense_shape)	
	
	
#Function for read and decode license plate information from a TFRecords File 
def read_and_decode(filename_queue,batch_size):

	reader =tf.TFRecordReader()
	_, serialized_data = reader.read(filename_queue)

	#First let us define how to parse our data
	context_features = {
		"length": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
		"input": tf.FixedLenSequenceFeature([150,], dtype=tf.int64),
		"label": tf.FixedLenSequenceFeature([9,], dtype=tf.int64)
	}
	
	# Now we parse the example
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		serialized=serialized_data,
		context_features=context_features,
		sequence_features=sequence_features
	)

	
	sequence_lengths,batched_data,batched_data_label = tf.train.batch(
		tensors=[ context_parsed["length"],sequence_parsed["input"],sequence_parsed["label"]],
		batch_size=batch_size,
		dynamic_pad=True
	)
	
	batched_data=tf.cast(batched_data,dtype=tf.float32)
	
	batched_data_label=tf.cast(batched_data_label,dtype=tf.int32)

	sequence_lengths=tf.cast(sequence_lengths,dtype=tf.int64)
	
	return(batched_data,batched_data_label,sequence_lengths)


	
	