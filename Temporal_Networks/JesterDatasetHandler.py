import numpy as np 
import os
import glob
import random
import cv2

def crop_center(img, new_height, new_width):
    orig_height, orig_width, _ = img.shape
    #orig_height, orig_width = img.shape
    startx = (orig_width//2) - (new_width//2)
    starty = (orig_height//2) - (new_height//2)
    return img[starty:starty+new_height, startx:startx+new_width]


def resize_image(img, new_height, new_width):
    h, w, _ = img.shape
    #h, w = img.shape
    if (h < new_height or w < new_width):
        img_data_r = imresize(img, (new_height, new_width))
    else:
        img_data_r = crop_center(img, new_height, new_width)
    return img_data_r

# (?) - MEAN SUBTRACT?
def handle_greyscale(img):
    img = img[:,:,0]
    #img = np.expand_dims(img, axis=0)
    return img

def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]

def create_oflow_stack(seq):
	# Given seq of ordered jpgs for the optical flow, read them into a numpy array
	# seq = [ \path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ... ]

	#print "in create oflow stack"
	oflow_stack = np.zeros(shape=(len(seq),100,100))

	# For each of the images in the sequence (which are contiguous optical flow images)
	for i in range(len(seq)):
		
		# Read the image as a color image (BGR) into a numpy array as 32 bit floats
		of_img = cv2.imread(seq[i]).astype(np.float32)
		#print "Shape after reading in : {}".format(of_img.shape)
		
		# Resize the image to 3x100x100
		of_img = resize_image(of_img, 100, 100)
		#print "Shape after Resizing : {}".format(of_img.shape)
		
		of_img = handle_greyscale(of_img)
		#print "Shape after greyscale : {}".format(of_img.shape)
	
		of_img /= 255.

		oflow_stack[i,:,:] = of_img

		#print "Printing"
		#print oflow_stack
	return oflow_stack
	#exit()


# Given the input dictionary file, we can make a list of sequences with labels that represent all
#	of the unique optical flow stacks that can be created from the input dictionary
#	The return value is a list of labeled sequences, each subsequence is length 20
def make_list_of_seqs(ifile,seq_size):

	my_list_of_seqs = []

	# Open the input dictionary file for reading
	# Each line contains a path to a directory full of jpgs that correspond
	#   to ONE video and the integer label representing the class for that video
	infile = open(ifile,"rb")

	# ex. line = "/.../jester/20bn-jester-v1/12 5"
	for line in infile:

	    split_line = line.split()

	    # Extract the class from the line
	    label = split_line[1].rstrip()

	    # Extract the path from the line
	    path = split_line[0]

	    # Change the path from the original 20bn-jester-v1 to 20bn-jester-v1-oflow
	    #path = path.replace("20bn-jester-v1","20bn-jester-v1-oflow-tvl1-6FPS-scaled")
	    path = path.replace("20bn-jester-v1","20bn-jester-v1-oflow-tvl1-6FPS-scaled")

	    assert(os.path.exists(path) == True)
	    
	    # Go into each directory and get an array of jpgs in the directory (these are full paths)
	    # Note: we only grab _h here, but we assume the _v exists
	    full_oflow_arr = glob.glob(path + "/*_h.jpg")

	    # Sort the array based on the sequence number [e.x. oflow_00028_00030_13_h.jpg ; seq# = 13]
	    full_oflow_arr.sort(key=lambda x: int(x.split("_")[-2]))

	    # Add the subsequences of length seq_size (usually 10)
	    # for   i < (len(arr) - 10)
	    for i in range(len(full_oflow_arr)-seq_size+1):
	        # Alloc list to store a single sequence of length seq_size
	        single_seq = []
	        # for j < 10
	        for j in range(seq_size):
	            # Append the horizontal version
	            single_seq.append(full_oflow_arr[i+j])
	            # Append the vertical version
	            single_seq.append(full_oflow_arr[i+j].replace("_h.jpg","_v.jpg"))
	        # Add this single sequence to the global list of sequences and the label
	        my_list_of_seqs.append([single_seq, label])

	# randomly shuffle list of contiguous sequences
	random.shuffle(my_list_of_seqs)

	# print total number of sequences
	num_sequences = len(my_list_of_seqs)
	print "Total number of sequences: {}".format(num_sequences)
	print "Finished creating list of sequences!"

	return my_list_of_seqs

class Jester_Dataset(object):
	def __init__(self, dictionary_file,seq_size=10):
		#self.video_dirs = [line.split()[0] for line in open(dictionary_file)]
		#self.labels = [line.split()[1] for line in open(dictionary_file)]
		self.list_of_seqs = make_list_of_seqs(dictionary_file,seq_size)
   	
   	def __getitem__(self, index):
		single_oflow_seq = self.list_of_seqs[index]
		return single_oflow_seq
    
	def __len__(self):
		return len(self.list_of_seqs)

	def read(self, batch_size=50, shuffle=True):

		"""Read (image, label) pairs in batch"""

		# Must make the order list a multiple of the batch size so we dont have problems
		num = int(len(self)) // batch_size
		num_batches = batch_size * num

		order = list(range(num_batches))

		if shuffle:
			random.shuffle(order)

	    # batch is a list of indexes, with length batch size
	    # i.e. bsize = 4 ; batch = [1,2,3,4] then [5,6,7,8] then [9,10,11,12]
		for batch in make_batch(order, batch_size):

			#print "Current Batch : {}".format(batch)
			oflow_batch, labels = [], []
			lseq = []

			for index in batch:

				# Single Seq = [ [\path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ...], 1 ]
				single_seq = self[index]

				# Extract the sequence of .jpgs
				seq = single_seq[0]
				lseq.append(seq)

				# Extract the label
				label = int(single_seq[1])

				# Create an optical flow stack from the seq images
				# oflow_stack is a np.ndarray with shape (20,100,100)
				oflow_stack = create_oflow_stack(seq)

				oflow_batch.append(oflow_stack)
				labels.append(label)

			yield np.stack(oflow_batch).astype(np.float32), np.stack(labels).astype(np.int32).reshape((batch_size,)), lseq
