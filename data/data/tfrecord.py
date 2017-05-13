import tensorflow as tf
import os
import cv2
import numpy as np


#Tfrecord File
tfrecords_file="train.tfrecords"
#Parameters
batch_size=10

#Functions used to structure our data for storage in a tfrecord file
def _int64_length_context_feature(value):
    return tf.train.Features(feature={'length':tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))})
    
def _int64_data_input_feature(image,length):
    return tf.train.FeatureList(feature=
	[tf.train.Feature(int64_list=tf.train.Int64List(value=[col_pix for col_pix in 
	(np.transpose(image[:,col,:]).flatten().tolist())]))for col in range(length)])
	
def _int64_data_input_label_feature(label):
    return tf.train.FeatureList(feature=
	[tf.train.Feature(int64_list=tf.train.Int64List(value=[prob_label for prob_label in 
	label[col]]))for col in range(9)])

	
	
def label_producer(label):
	full_label=[]
	label_length=len(label)
	for char in label : 
		index_label=np.zeros(37,dtype=int)
		#Dealing with number characters
		if(ord(char)<58 and ord(char)>47):
			index=ord(char)-48
			np.put(index_label,index,1)
			full_label.append(index_label)
		#Dealing with uppercase characters
		if(ord(char)<91 and ord(char)>64):
			index=ord(char)-55
			np.put(index_label,index,1)
			full_label.append(index_label)
		#Dealing with lowercase characters
		if(ord(char)<123 and ord(char)>96):
			index=ord(char)-87
			np.put(index_label,index,1)
			full_label.append(index_label)
		#Dealing with "-" character
		if(ord(char)==45):
			index=36
			np.put(index_label,index,1)
			full_label.append(index_label)
	return(full_label)
	
def recon_label(label):
	#Our final string plate from the hot vector
	full_label=[]
	#First get the number of characters in the plate
	label_length=label.shape[0]
	#Now get the maximum index of each line and get the character it corresponds to
	for line in range(label_length):
		max_ind=np.argmax(label[line])
		#Dealing with number characters
		if(max_ind<10):
			character=chr(max_ind+48)
			full_label.append(character)
		#Dealing with characters
		if(max_ind>9 and max_ind<36):
			character=chr((max_ind+55))
			full_label.append(character)
		#Dealing with '-' character
		if(max_ind==36):
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
    return(buffer/255)


	
def create_tfrecord_file(tfrecords_file):
	writer=tf.python_io.TFRecordWriter(tfrecords_file)

	#First we get all of the files of the directory
	file_list=os.listdir()
	np.random.shuffle(file_list)
	#Then we get the necessary information (license plate number, sequence length(image-width) and image )
	for file in file_list :
		if(file.endswith('png')):
		
			image=cv2.imread(file)
			sequence_length=image.shape[1]
			label=file.split(".")[0]
			plate=label_producer(label)
			
			example=tf.train.SequenceExample(
				context=_int64_length_context_feature(sequence_length),
				feature_lists=(tf.train.FeatureLists(
							feature_list={
							'input':_int64_data_input_feature(image,sequence_length),
							'label':_int64_data_input_label_feature(plate)}
			)))
			# print(example.ListFields())
			writer.write(example.SerializeToString())
		
	writer.close()

#Function for read and decode license plate information from a TFRecords File 
def read_and_decode(filename_queue):

	reader =tf.TFRecordReader()
	_, serialized_data = reader.read(filename_queue)

	#First let us define how to parse our data
	context_features = {
		"length": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
		"input": tf.FixedLenSequenceFeature([150,], dtype=tf.int64),
		"label": tf.FixedLenSequenceFeature([37,], dtype=tf.int64)
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
	print("\n"*10)
	results = tf.contrib.learn.run_n({"seq_len":sequence_lengths,"input_batch":batched_data,"input_label":batched_data_label}, n=1, feed_dict=None)

	images=results[0]["input_batch"]

	labels=results[0]["input_label"]
	
	lengths=results[0]["seq_len"]

	
	max_len=np.amax(lengths)
	for i in range(batch_size):
		recon=recon_image(images[i],max_len)
		cv2.imshow(recon_label(labels[i]),recon)
		
	cv2.waitKey(0)
	
	return(batched_data,batched_data_label,sequence_lengths)

create_tfrecord_file(tfrecords_file)

###########################
#TESTING OUR TFRECORD FILE#
###########################

#Creation of a queue, working with 10 epochs so 10*100 images, an image will basically be shown 10 times
filename_queue=tf.train.string_input_producer([tfrecords_file],num_epochs=None)
#Get an image batch
image_batch,label_batch,sequence_length=read_and_decode(filename_queue)