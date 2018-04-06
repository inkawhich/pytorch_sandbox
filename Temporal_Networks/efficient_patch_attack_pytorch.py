from matplotlib import pyplot as plt
import numpy as np
import os
import operator
import JesterDatasetHandler2 as jdh
import cv2
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import MyResnet as myresnet 

##################################################################################
# Function to calculate dense optical flow between two adjacent frames
##################################################################################
def calc_optical_flow(frame1, frame2):

    frame1 = frame1.astype(np.uint8)
    frame2 = frame2.astype(np.uint8)

    # Convert the images to grayscale
    f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    ### Dual TV-L1
    #oflow_tvl1= cv2.DualTVL1OpticalFlow_create()
    #flow = oflow_tvl1.calc(f1_gray, f2_gray, None)
    ### Farneback
    flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Separate horizontal and vertical pieces
    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    ### Perform Scaling
    # Clip based on evaluation of distribution
    # This range should include about 99.9% of values
    MAX_V=16
    MIN_V=-16
    # Clip range
    h_oflow[h_oflow < MIN_V] = MIN_V
    h_oflow[h_oflow > MAX_V] = MAX_V
    v_oflow[v_oflow < MIN_V] = MIN_V
    v_oflow[v_oflow > MAX_V] = MAX_V
    # Scale the space [MIN_V,MAX_V] to [0,255]
    OldMax = MAX_V
    OldMin = MIN_V
    NewMax = 255
    NewMin = 0
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    h_oflow = (((h_oflow - OldMin) * NewRange) / OldRange) + NewMin
    v_oflow = (((v_oflow - OldMin) * NewRange) / OldRange) + NewMin

    # Simulate the jpg writing by rounding to int
    h_oflow = np.rint(h_oflow)
    v_oflow = np.rint(v_oflow)

    # Return the oflow
    return h_oflow,v_oflow

##################################################################################
# Recompute full oflow stack from image array
##################################################################################
def compute_full_optical_flow_stack(imgs, height, width):
    # Allocate space to put newly calculated optical flow
    new_pstack = np.zeros((1,20,height,width))

    # For every pair of images, calculate optical flow and store it in the new stack
    cnt = 0
    for i in range(0,10):
        tmph,tmpv = calc_optical_flow(imgs[i].astype(np.uint16), imgs[i+1].astype(np.uint16))
        new_pstack[0,cnt] = tmph
        new_pstack[0,cnt+1] = tmpv
        cnt += 2   

    new_pstack /= 255.
    new_pstack[ new_pstack<0 ] = 0
    new_pstack[ new_pstack>1 ] = 1

    return new_pstack

##################################################################################
### Get the original images given the optical flow image paths
##################################################################################
# oflow_seq = [\path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ...]
def get_images_from_oflow_seq( oflow_seq, height, width ):
    
    seq = oflow_seq[::2]
    #print seq

    #prefix_path = os.path.dirname(seq[0]).replace("20bn-jester-v1-alt-10class-oflow-tvl1-6FPS-scaled16-150x100","20bn-jester-v1")
    prefix_path = os.path.dirname(seq[0]).replace("20bn-jester-v1-alt-10class-oflow-farneback-6FPS-scaled16-150x100","20bn-jester-v1")

    # List to store the filenames of the images we want, these are derived from the file names of the 
    #   optical flow paths that are input in oflow_seq
    # fnames = = [ '/full/path/to/00001.jpg', '/full/path/to/00002.jpg', '/full/path/to/00003.jpg', ...]
    fnames = []
    for file in seq:
        oflow_file = os.path.basename(file)
        i1 = prefix_path + "/" + oflow_file.split("_")[1] + ".jpg"
        i2 = prefix_path + "/" + oflow_file.split("_")[2] + ".jpg"
        if i1 not in fnames:
            fnames.append(i1)
        if i2 not in fnames:
            fnames.append(i2)

    assert(len(fnames) == 11)

    images = np.zeros((11,height,width,3))
    for i,f in enumerate(fnames):
        # [0,255]
        of_img = cv2.imread(f).astype(np.float32)
        of_img = jdh.resize_image(of_img, height, width)
        #of_img = jdh.crop_center(of_img, height, width)
        #of_img = of_img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
        images[i,:,:,:] = of_img

    return images

##################################################################################
### Classify single oflow stack and return prediction, confidence, and softmax arr
##################################################################################
def classify_stack(model, stk, lbl):

    ##### PYTORCH STUFF
    # Create the data and label variables so we can use them in the computation
    img_var = Variable(torch.FloatTensor(stk),requires_grad=False)

    # Call a forward pass on the data
    output = model(img_var)

    # Get the max index from the ith softmax array
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    #softmax = output.cpu().data.numpy()
    softmax = output.data

    s = nn.Softmax()
    smax = s(Variable(softmax))
    conf = smax.cpu().data.numpy()[0][pred]

    #print "pred: ", int(pred)
    #print "conf: ", float(conf)
    #print "smax: ", smax.cpu().data.numpy()
    #exit()

    return int(pred),float(conf),smax.cpu().data.numpy()

##################################################################################
### Place square of color c on img at loc x,y
##################################################################################
def perturb_loc(img, x, y, c, N, max_x, max_y):
    # Place cluster of N pixels with color c onto img at loc x,y
    for i in range(N):
        for j in range(N):
            if (((x+i) > 0) and ((x+i) < max_x) and ((y+j) > 0) and ((y+j) < max_y)):
                img[x+i,y+j] = c
    return img

##################################################################################
### Compound Method. Perturb imgs, compute stack, reclassify stack, return softmax
##################################################################################
# Given a full image array, place a square perturbation on img_arr[img_index] at location (x_loc,y_loc)
#   with RGB color pert_color and width pert_width. The max_* inputs are so we do not perturb out
#   of bounds. Once the perturbation is applied to the image, the optical flow stack is recomputed
#   from the newly perturbed image array and the stack is classified by the model. This function returns
#   the softmax array of the stack and a copy of the perturbed image array.
def test_perturbation(input_oflow_stack, img_arr, img_index, x_loc, y_loc, pert_color, pert_width, max_x, max_y, model):
    
    # tmp_ia = copy of image array so we dont screw anything up
    tmp_ia = np.copy(img_arr)

    # move the perturbation to the right by mv units on cpy_img_arr[i]
    tmp_ia[img_index] = perturb_loc(tmp_ia[img_index], x_loc, y_loc, pert_color, pert_width, max_x, max_y)

    # calc optical flow stack
    #delete_of = compute_full_optical_flow_stack(tmp_ia,max_x,max_y)

    # Calculate relevant oflows between frames that were effected by the perturbation
    #   and place them into the input_oflow_stack at the correct position rather
    #   than recomputing the whole stack every time.
    tmp_of = np.copy(input_oflow_stack)

    # Recompute oflow (_h, _v) between tmp_ia[img_index-1] && tmp_ia[img_index]
    tmph,tmpv = calc_optical_flow(tmp_ia[img_index-1].astype(np.uint16), tmp_ia[img_index].astype(np.uint16))
    tmph = tmph / 255.
    tmpv = tmpv / 255.
    tmph[ tmph<0 ] = 0
    tmph[ tmph>1 ] = 1
    tmpv[ tmpv<0 ] = 0
    tmpv[ tmpv>1 ] = 1
    # Place _h at tmp_of[2*(img_index-1)]
    tmp_of[0,2*(img_index-1)] = tmph

    # Place _v at tmp_of[2*(img_index-1) + 1]
    tmp_of[0,2*(img_index-1) + 1] = tmpv

    # If img_index < 10
    if img_index < 10:

        # Recompute oflow (_h, _v) between tmp_ia[img_index] && tmp_ia[img_index+1]
        tmph,tmpv = calc_optical_flow(tmp_ia[img_index].astype(np.uint16), tmp_ia[img_index+1].astype(np.uint16))
        tmph = tmph / 255.
        tmpv = tmpv / 255.
        tmph[ tmph<0 ] = 0
        tmph[ tmph>1 ] = 1
        tmpv[ tmpv<0 ] = 0
        tmpv[ tmpv>1 ] = 1
        # Place _h at tmp_of[2*img_index]
        tmp_of[0,2*img_index] = tmph
        
        # Place _v at tmp_of[2*img_index + 1]
        tmp_of[0,2*img_index + 1] = tmpv


    # conf_right = run inference on optical flow stack and get new confidence
    _,_,tsmax = classify_stack(model, tmp_of, 0)
    
    return tsmax, tmp_ia

##################################################################################
### Print an image array
##################################################################################
def print_image_array(imgs):

    #if imgs.max() > 1:
    imgs /= 255.

    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.axis("off")
        new = imgs[i]
        plt.imshow(new[:,:,(2,1,0)])
    plt.show()

##################################################################################
### Print the images and their corresponding saliency oflow maps
##################################################################################
def print_new_and_old(orig_imgs, pert_imgs, orig_oflow, pert_oflow):

    print orig_imgs.shape

    orig_imgs /= 255.
    pert_imgs /= 255.

    ocnt = 0

    # Print the original images
    for i in range(10):
        plt.subplot(8,10,i+1)
        plt.axis("off")
        plt.imshow(orig_imgs[i][:,:,(2,1,0)])

    # Print the ORIGINAL oflow
    # Horizontal first
    cnt=0
    for i in range(10):
        plt.subplot(8,10,i+1+10)
        plt.axis("off")
        plt.imshow(orig_oflow[cnt], vmin=0, vmax=1., cmap='gray')
        cnt += 2
    # Vertical here
    cnt=1
    for i in range(10):
        plt.subplot(8,10,i+1+20)
        plt.axis("off")
        plt.imshow(orig_oflow[cnt], vmin=0, vmax=1., cmap='gray')
        cnt += 2

    # Magnitude here
    #compute magnitude of oflow pairs
    for k,i in enumerate(range(0,20,2)):
        mag = np.sqrt(orig_oflow[i]**2 + orig_oflow[i+1]**2)
        plt.subplot(8,10,k+1+30)
        plt.axis("off")
        plt.imshow(mag, vmin=0, vmax=1.5, cmap='gray')

    # Print the CURRENT PERTURBED oflow
    # Horizontal first
    cnt=0
    for i in range(10):
        plt.subplot(8,10,i+1+40)
        plt.axis("off")
        plt.imshow(pert_oflow[cnt], vmin=0, vmax=1., cmap='gray')
        cnt += 2
    # Vertical here
    cnt=1
    for i in range(10):
        plt.subplot(8,10,i+1+50)
        plt.axis("off")
        plt.imshow(pert_oflow[cnt], vmin=0, vmax=1., cmap='gray')
        cnt += 2

    # Magnitude here
    #compute magnitude of oflow pairs
    for k,i in enumerate(range(0,20,2)):
        mag = np.sqrt(pert_oflow[i]**2 + pert_oflow[i+1]**2)
        plt.subplot(8,10,k+1+60)
        plt.axis("off")
        plt.imshow(mag, vmin=0, vmax=1.5, cmap='gray')

    # Print the current images
    for i in range(10):
        plt.subplot(8,10,i+1+70)
        plt.axis("off")
        plt.imshow(pert_imgs[i][:,:,(2,1,0)])


    plt.show()

##################################################################################
# L-0 Norm of videos
##################################################################################
def compute_number_of_locations_perturbed(original_img_arr, perturbed_img_arr):
    diff_video = original_img_arr[:,:,:,1] - perturbed_img_arr[:,:,:,1]
    print diff_video.shape
    non_zero_count = np.count_nonzero(diff_video)
    return non_zero_count

##################################################################################
# compute_stdev_map
##################################################################################
# This function computes the standard deviation map called for
#   in the paper. It computes the standard deviation of each
#   n x n neighborhood in the input img.
#   Note: the img should have a border on it so the stdev map
#       has the same dimensionality as the original image
def compute_stdev_map(img, n):

    stdev_map = np.zeros(shape=[ (img.shape[0] - (n-1)), (img.shape[1]-(n-1)) ]) 
    for i in range(img.shape[0] - (n-1)):
        for j in range(img.shape[1] - (n-1)):
            #print "img[{},{}] = \n{}".format(i,j,img[i:i+n, j:j+n])
            #print "avg = {}".format(np.average(img[i:i+n, j:j+n]))
            #print "std = {}".format(np.std(img[i:i+n, j:j+n]))
            #print 
            stdev_map[i,j] = np.std(img[i:i+n, j:j+n])

    return stdev_map 

##################################################################################
# apply_border
##################################################################################
# This function returns a new image which is the input image
#   img with a n-pixel border of zeros around it. 
def apply_border(img, n):
    
    # Initialize a new matrix of zeros the size of the img plus the border
    new_img = np.zeros(shape=[ (img.shape[0]+2*n) , (img.shape[1]+2*n) ])

    # add the old image onto the top of the new zeros matrix
    new_img[n:n+img.shape[0], n:n+img.shape[1]] += img

    return new_img

##################################################################################
# Get coords of N smallest values in the 2d matrix
##################################################################################
def get_coords_of_n_smallest(mat, n, low_var=True):
    # Assert mat is a 2d matrix
    assert(len(mat.shape) == 2)
    flat = []
    #for i in range(mat.shape[0]):
    #    for j in range(mat.shape[1]):
    #        flat.append([i,j,mat[i,j]])
    for i in range(5,mat.shape[0]-5):
        for j in range(5,mat.shape[1]-5):
            flat.append([i,j,mat[i,j]])
    #print flat
    # Note: each entry in flat is [x_coord, y_coord, value]
    # sort the flat list according to the value at that coordinate
    if low_var:
        flat.sort(key=lambda x : x[2],reverse=True)
    else:
        flat.sort(key=lambda x : x[2],reverse=False)
    # Extract the n_largest entries from the flat list
    n_largest = flat[-n:]
    # Just take the coordinates of the n_largest, dont care about the values here
    coords = [x[0:2] for x in n_largest]
    return coords 

##################################################################################
##################################################################################   
##################################################################################
### INPUTS
##################################################################################
##################################################################################   
##################################################################################
# Model inputs
TEST_DICT = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'alt_TestDictionary_10class.txt')
#SAVED_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'pytorch_sandbox', 'Temporal_Networks', 'resnet50_alt_10class_saved_model_110000.pth')
SAVED_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'pytorch_sandbox', 'Temporal_Networks', 'resnet50_ft_farneback_alt_10class_saved_model_115000.pth')
GPU = 1

# Attack specific inputs
BOX_WIDTH = 2  # width of perturbation patch placed on image
BOX_STRIDE = 2  # how much to move a perturbation at any given step
MAX_PERTS = 20  # max number of pert. sequences to apply to a single stack
NUM_VARIANCE_COORDS = 200 # when using variance aware seeding, how many of the lowest variance locs to consider

# General inputs
NUM_CLASSES = 10 # Number of classes the model is trained on
HEIGHT = 100
WIDTH = 150

##################################################################################
### Bring up network & Initialize Dataset
##################################################################################
# Make sure the specified inputs exist
if ((not os.path.exists(TEST_DICT)) or (not os.path.exists(SAVED_NET))):
    print ("ERROR: An input was not found")
    exit()

# Initialize the model
# Custom Resnet50 (the [3,4,6,3] is what makes it Resnet50. Look at orig pytorch resnet.py for other configs)
test_model = myresnet.ResNet(myresnet.Bottleneck, [3, 4, 6, 3], input_depth=20, num_classes=NUM_CLASSES)

# For GPU support
if GPU:
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    test_model = torch.nn.DataParallel(test_model).cuda()

# Load the model from the checkpoint
test_model = torch.load(SAVED_NET)
# Set it to test mode (for batch norm and dropout layers)
test_model.eval()

# Initialize Dataset Object
test_dataset = jdh.Jester_Dataset(dictionary_file=TEST_DICT,seq_size=10)


##################################################################################
### Start running tests
##################################################################################
# Keeping stats
num_correct = 0
num_succeed = 0
num_attacks = 0
total = 0

###############################
# Grab a fresh stack
###############################
# For all of the stacks parsed in from the file
for stack, label, seq in test_dataset.read(batch_size=1,shuffle=False):

    # Extract the ground truth label
    gt_label = label[0]

    print "\n\n"
    print "**************************************"
    print "Grabbing new oflow stack"
    print "**************************************"
    print seq
    print "Ground Truth Label: ", gt_label

    # Flag used when breaking out of inner for-loops
    adv = 0

    total += 1

    # Classify the stack to get the initial prediction
    pred,_,smx = classify_stack(test_model, stack, label)
    gtcnf = smx[0][gt_label]
    print smx[0]
    print "initial GT confidence: ", gtcnf

    # If initial prediction is incorrect, no need to attack
    if (pred != label[0]):
        print("INCORRECT INITIAL PREDICTION; SKIPPING IMAGE")
        continue
    
    ##### If here, initial pred was correct, we will now attempt to generate adversarial example

    # if initial prediction is correct, increment num_attacks
    num_attacks += 1

    # Get original images for the stack (img_arr)
    Img_arr = get_images_from_oflow_seq(seq[0], HEIGHT, WIDTH)
    #print_image_array(np.copy(Img_arr))
    print Img_arr.shape

    # Make a copy of the Img_arr to play with here
    tmp_img_arr = np.copy(Img_arr)

    # Sanity check. Make sure this prediction is the same as the initial [It should be!]. The original prediction
    #   was made on the images that were loaded in from disk, this prediction represents our recaluclated
    #   optical flow stack and should obviously be the same.
    garbage_stack = compute_full_optical_flow_stack(tmp_img_arr,HEIGHT,WIDTH)
    garbage_pred,_,garbage_smax = classify_stack(test_model, garbage_stack, label)
    print "redundant prediction: ", garbage_pred
    print "redundant smax: ", garbage_smax

    # Look at the difference between the 'stack' that was read in from the saved images and
    #   the recomputed optical flow. [Hint: There should be NO difference!]
    print "orig stack stuff"
    print "\tmin: ",stack.min(axis=(0,1,2,3))
    print "\tmax: ",stack.max(axis=(0,1,2,3))
    print "\tavg: ",stack.mean(axis=(0,1,2,3))
    print "recomputed stuff"
    print "\tmin: ",garbage_stack.min(axis=(0,1,2,3))
    print "\tmax: ",garbage_stack.max(axis=(0,1,2,3))
    print "\tavg: ",garbage_stack.mean(axis=(0,1,2,3))
    
    # Assert the prediction made on the original stack created with the jpgs is the same as
    #   the prediction of the recomputed but unaltered stack.
    assert(pred == garbage_pred)

    ###### For variance seeding
    # Compute stdev map here to get candidate locations for seeding perturbations
    first_frame = np.copy(tmp_img_arr[0])
    gray_first_frame = (first_frame[:,:,0] + first_frame[:,:,1] + first_frame[:,:,2])/3.
    bordered_first_frame = apply_border(gray_first_frame, 1)
    first_frame_stdev_map = compute_stdev_map(bordered_first_frame, 3)
    low_variance_locs = get_coords_of_n_smallest(first_frame_stdev_map, NUM_VARIANCE_COORDS,low_var=False)

    # Keep track of the last 'best' image array and confidence. This is kept track of because after
    #  we apply a perturbation sequence, if the resulting image array is less desirable we can 
    #  just ditch it and go back to the last best version. This will make sure that we are always
    #  moving towards our objective.
    last_useful_img_arr = np.copy(Img_arr)
    last_useful_conf = gtcnf

    ###############################
    # Start a perturbation sequence
    ###############################

    # For how many perturbations we want to apply
    for i in range(MAX_PERTS):
        
        # If we are here because we broke out of the inner for loop, break again
        if adv == 1:
            break

        # Check if the last perturbation sequence bought us anything
        # if it did NOT, we dont want to keep the sequence so we revert to the saved image array from last iter
        check_prev_iter_stack = compute_full_optical_flow_stack(tmp_img_arr,HEIGHT,WIDTH)
        _,_,prev_iter_smax = classify_stack(test_model, check_prev_iter_stack, label)
        prev_iter_conf = prev_iter_smax[0][label]

        # If the prev iter did not decrease the last best confidence, the sequence is useless so get rid of it
        if prev_iter_conf >= last_useful_conf:
            print "last iter was useless, reverting"
            tmp_img_arr = np.copy(last_useful_img_arr)
        # If it did improve then we will reset our baseline
        else:
            print "last iter was GOOD, keeping it"
            last_useful_img_arr = np.copy(tmp_img_arr)
            last_useful_conf = prev_iter_conf

        print "\nStarting new perturbation sequence..."

        ####################################################################################
        ################### SELECT AN INITIAL LOCATION FOR THIS SEQUENCE ###################
        ####################################################################################
        ### Option 1: loc = Pick a random x,y loc inside 100x100
        #x_coord = random.randint(BOX_WIDTH,HEIGHT-BOX_WIDTH)
        #y_coord = random.randint(BOX_WIDTH,WIDTH-BOX_WIDTH)

        ### Option 2: Variance Aware seeding
        low_variance_loc_index = random.randint(0,NUM_VARIANCE_COORDS-1)
        x_coord,y_coord = low_variance_locs[low_variance_loc_index]

        ### Option 3: Manually select start loc
        #print_image_array(np.copy(tmp_img_arr))
        #plt.imshow(np.copy(tmp_img_arr)[0][:,:,(2,1,0)]/255.)
        #plt.show()
        #x_coord = int(input("Enter x_coord: "))
        #y_coord = int(input("Enter y_coord: "))
        ####################################################################################

        ##### Pick random RGB color
        pcolor = random.sample(range(1, 255), 3)

        # Apply perturbation to img_arr[0] at (x,y) with rand color
        tmp_img_arr[0] = perturb_loc(tmp_img_arr[0], x_coord, y_coord, pcolor, BOX_WIDTH, HEIGHT, WIDTH)

        # Calc oflow stack
        tmp_oflow_stack = compute_full_optical_flow_stack(tmp_img_arr,HEIGHT,WIDTH)

        # curr_conf = run inference on oflow stack to get baseline confidence
        init_pred,_,init_smax = classify_stack(test_model, tmp_oflow_stack, label)
        print "initial prediction: ", init_pred
        #print "initial confidence: ", init_conf

        # Check if this single perturbation caused a misclassification
        if (i == 0) and (init_pred != label[0]):
            adv = 1
            num_succeed += 1
            print "***************EARLY SUCCESS!***************"
            print "Ground Truth: ", label[0]
            print "Prediction: ", init_pred
            print "Confidence: ", init_smax[0][init_pred]
            break

        print init_smax[0]
        init_conf = init_smax[0][gt_label]
        print "GT confidence: ", init_conf

        # For this perturbation sequence we will save the image arrays created at each step
        #   along with the associated confidences. When we have finished the sequence, we will
        #   use the image array that had the most desirable confidence score as the representative
        #   image array of that iteration. We will then check if that image array is better
        #   than the previous best image array.
        saved_image_arrays = []
        saved_confidence_scores = []

        ###############################
        # Test and perturb each frame
        ###############################

        for img_arr_index in range(1,len(tmp_img_arr)):
            
            # Differences incurred by  [no-move, moving right, left, up, down]
            diffs = [float("-inf"),float("-inf"),float("-inf"),float("-inf"),float("-inf")]
            # Save the new image arrays so we dont have to recompute the best one at the end
            new_image_arrays = [np.zeros(shape=Img_arr.shape),np.zeros(shape=Img_arr.shape),np.zeros(shape=Img_arr.shape),np.zeros(shape=Img_arr.shape),np.zeros(shape=Img_arr.shape)]

            # Compute Optical Flow Stack Here And Pass it to all of the test perturbation functions. Since the test perturbation functions
            #  only modify one image at a time, it is unnecessary for them to recompute the entire stack every single time. Rather, 
            #  they only have to recompute the optical flow that is relevant to the single image being modified.
            current_iters_oflow_stack = compute_full_optical_flow_stack(tmp_img_arr,HEIGHT,WIDTH)

            ####### NO MOVEMENT
            # Test no movement as baseline
            if (x_coord) < HEIGHT:
                tsmax,tmp_ia = test_perturbation(current_iters_oflow_stack, tmp_img_arr, img_arr_index, x_coord, y_coord, pcolor, BOX_WIDTH, HEIGHT, WIDTH, test_model)
                # Extract the current confidence of the ground truth label
                tconf = tsmax[0][gt_label]
                # Save the amount that the perturbation changes the confidence
                diffs[0] = init_conf - tconf
                # Also save the perturbed image array that was generated from this perturbation. This is saved so 
                #   we dont have to recompute it when we pick the best perturbation
                new_image_arrays[0] = tmp_ia
           
            ####### LOOK RIGHT
            # If the box can move right
            if (y_coord + BOX_STRIDE) < (WIDTH-BOX_WIDTH):
                tsmax,tmp_ia = test_perturbation(current_iters_oflow_stack, tmp_img_arr, img_arr_index, x_coord, y_coord+BOX_STRIDE, pcolor, BOX_WIDTH, HEIGHT, WIDTH, test_model)
                # Extract the current confidence of the ground truth label
                tconf = tsmax[0][gt_label]
                # Save the amount that the perturbation changes the confidence
                diffs[1] = init_conf - tconf
                # Also save the perturbed image array that was generated from this perturbation. This is saved so 
                #   we dont have to recompute it when we pick the best perturbation
                new_image_arrays[1] = tmp_ia
            
            ####### LOOK LEFT
            # If the box can move right
            if (y_coord - BOX_STRIDE) > 0:
                tsmax,tmp_ia = test_perturbation(current_iters_oflow_stack, tmp_img_arr, img_arr_index, x_coord, y_coord-BOX_STRIDE, pcolor, BOX_WIDTH, HEIGHT, WIDTH, test_model)
                # Extract the current confidence of the ground truth label
                tconf = tsmax[0][gt_label]
                # Save the amount that the perturbation changes the confidence
                diffs[2] = init_conf - tconf
                # Also save the perturbed image array that was generated from this perturbation. This is saved so 
                #   we dont have to recompute it when we pick the best perturbation
                new_image_arrays[2] = tmp_ia
           
            ####### LOOK UP
            # If the box can move up
            if (x_coord - BOX_STRIDE) > 0:
                tsmax,tmp_ia = test_perturbation(current_iters_oflow_stack, tmp_img_arr, img_arr_index, x_coord-BOX_STRIDE, y_coord, pcolor, BOX_WIDTH, HEIGHT, WIDTH, test_model)
                # Extract the current confidence of the ground truth label
                tconf = tsmax[0][gt_label]
                # Save the amount that the perturbation changes the confidence
                diffs[3] = init_conf - tconf
                # Also save the perturbed image array that was generated from this perturbation. This is saved so 
                #   we dont have to recompute it when we pick the best perturbation
                new_image_arrays[3] = tmp_ia
           
            ####### LOOK DOWN
            # If the box can move down
            if (x_coord + BOX_STRIDE) < (HEIGHT-BOX_WIDTH):
                tsmax,tmp_ia = test_perturbation(current_iters_oflow_stack, tmp_img_arr, img_arr_index, x_coord+BOX_STRIDE, y_coord, pcolor, BOX_WIDTH, HEIGHT, WIDTH, test_model)
                # Extract the current confidence of the ground truth label
                tconf = tsmax[0][gt_label]
                # Save the amount that the perturbation changes the confidence
                diffs[4] = init_conf - tconf
                # Also save the perturbed image array that was generated from this perturbation. This is saved so 
                #   we dont have to recompute it when we pick the best perturbation
                new_image_arrays[4] = tmp_ia
           
            ##### Look at what perturbation had the most impact

            # Which perturbation affected the confidence the most?
            best_dir, best_value = max(enumerate(diffs), key=operator.itemgetter(1))

            assert(best_value != float("-inf"))

            # Print summary of how moving the perturbation on this image changed the confidence
            print "Status: Pert Cnt = [{} / {}], Iter Cnt = [{} / {}]".format(i,MAX_PERTS,img_arr_index,10)
            print "Diffs <nm,r,l,u,d>: ", diffs
            print "best_dir: ", best_dir
            print "best_value: ", best_value

            # If none of the perts decreased the confidence we will not perturb on this frame
            if best_value <= 0:
                print "NO VALUE ADDED!"
               
            ##### Set up the variables for the next iteration

            # Apply the perturbation to img_arr[i] at the most sensitive spot
            tmp_img_arr = new_image_arrays[best_dir]

            # Construct the coordinate of the current perturbation to be used next iteration
            # If no-movement was the best Do nothing, but the box still gets fixed on the image
            if best_dir == 0:
                x_coord = x_coord
            # If right was the best direction
            elif best_dir == 1:
                y_coord += BOX_STRIDE
            # If left was the best direction
            elif best_dir == 2:
                y_coord -= BOX_STRIDE
            # If up was the best direction
            elif best_dir == 3:
                x_coord -= BOX_STRIDE
            # If down was the best direction
            elif best_dir == 4:
                x_coord += BOX_STRIDE
            else:
                print "BIG PROBLEMS"
                exit()

            # Make sure the new coord is not out-of-bounds
            if (x_coord<0):
                x_coord = 0
            if (x_coord>= HEIGHT):
                x_coord = HEIGHT-10
            if (y_coord<0):
                y_coord = 0 
            if (y_coord>= WIDTH):
                y_coord = WIDTH-10

            print "new coord: ", x_coord, " ",y_coord

            # conf = new confidence
            tmp_of = compute_full_optical_flow_stack(tmp_img_arr,HEIGHT,WIDTH)
            curr_pred,_,s = classify_stack(test_model, tmp_of, label)
            init_conf = s[0][gt_label]

            # Save the image arrays from this iteration because it might be better than the final one
            saved_image_arrays.append(np.copy(tmp_img_arr))
            saved_confidence_scores.append(init_conf)

            #print_new_and_old(np.copy(Img_arr), np.copy(tmp_img_arr), np.squeeze(np.copy(garbage_stack)), np.squeeze(np.copy(tmp_of)))

            print "new GT confidence:", init_conf

            # Check if the image array is adversarial
            if curr_pred != label[0]:
                print "\n\n***************SUCCESS BABY!***************"
                print s
                print "Ground Truth Label: ", label[0]
                print "Current GT Confidence: ", s[0][gt_label]
                print "Current Prediction: ", curr_pred
                print "Current Confidence: ", s[0][curr_pred]
                pert_loc_count = compute_number_of_locations_perturbed(np.copy(Img_arr), np.copy(tmp_img_arr))
                print "Number of locs perturbed: ", pert_loc_count
                print "Percent locs perturbed: ", pert_loc_count/float(WIDTH*HEIGHT*11)

                #print_new_and_old(np.copy(Img_arr), np.copy(tmp_img_arr), np.squeeze(np.copy(garbage_stack)), np.squeeze(np.copy(tmp_of)))

                num_succeed += 1
                #print_image_array(np.copy(tmp_img_arr))

                # Save the img array as a gif
                import imageio
                images = []
                for pimg in tmp_img_arr:
                    images.append(pimg[:,:,(2,1,0)])
                imageio.mimsave('~/shared_host/adversarial_results/adversarial_video.gif', images)

                exit()
                adv = 1
                break

        # We have evaluated moving the box across all of the images. Before we jump back up to start a new
        #   perturbation sequence, we will pick the image array from this run that had the absolute best
        #   confidence score to use as the representative image array for this run. This is important because
        #   at the start of next run we check if the best run from the previous iteration is better than the 
        #   global best run.

        # Find the index of the lowest conficence score for the run
        best_ind,lowest_pert_conf = min(enumerate(saved_confidence_scores), key=operator.itemgetter(1))
        tmp_img_arr = np.copy(saved_image_arrays[best_ind])

        print "\n\n"
        print "End of perturbation sequence: {}! Best Achieved Confidence: {}".format(i,lowest_pert_conf)
        print "\n"

    if adv == 0:
        print "FAIL!!"
        

print ("\n**************************************")
print ("Total Stacks Tested = ",total)
print ("Number of attacks attempted = ",num_attacks)
print ("Number of successful attacks = ",num_succeed)
print ("Old Accuracy = ",num_attacks/float(total))
print ("Adv Accuracy = ",(num_attacks-num_succeed)/float(total))




