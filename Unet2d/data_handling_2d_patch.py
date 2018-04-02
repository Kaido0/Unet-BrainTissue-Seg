
from __future__ import print_function

# import packages
import time, os, random
import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches

# import configurations
import configs

# init configs
image_rows = configs.VOLUME_ROWS
image_cols = configs.VOLUME_COLS
image_depth = configs.VOLUME_DEPS
num_classes = configs.NUM_CLASSES

# patch extraction parameters
patch_size = configs.PATCH_SIZE
extraction_step = configs.EXTRACTTION_STEP
extraction_step_csf_only = configs.EXTRACTTION_STEP_CSF

# create npy data
def create_npy_data(train_imgs_path, is_extract_more_csf, is_train):
    
    patches_training_imgs_2d=np.empty(shape=[0, patch_size,patch_size], dtype='int16')
    patches_training_gtruth_2d=np.empty(shape=[0, patch_size,patch_size, num_classes], dtype='int16')
    
    images_train_dir = os.listdir(train_imgs_path)
    start_time = time.time()

    j=0
    print('-'*30)
    print('Creating training 3d_patches...')
    print('-'*30)

    print('Hello')
    train_x=np.load('../npy_data/train_x.npy')
    train_y=np.load('../npy_data/train_y.npy')
    mask=np.load('../npy_data/train_mask.npy')
    
    # for each volume do:
    for i in range(6):
        patches_training_imgs_2d_temp = np.empty(shape=[0,patch_size,patch_size], dtype='int16')
        patches_training_gtruth_2d_temp = np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
        print('Processing: volume {0} / {1} volume images'.format(j+1, len(images_train_dir)))

        # load volume, gt and mask

        img_data = train_x[i]

        img_gtruth_data = train_y[i]
        
        img_mask_data = mask[i]
        
        # for each slice do
        for slice in [27,28,29]:
            print(slice)
            patches_training_imgs_2d_slice_temp = np.empty(shape=[0,patch_size,patch_size], dtype='int16')
            patches_training_gtruth_2d_slice_temp = np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
            if np.count_nonzero(img_gtruth_data[slice,:,:]) and np.count_nonzero(img_data[slice,:,:]):
        
                # extract patches of the jth volum image
                imgs_patches, gt_patches = extract_2d_patches(img_data[slice,:,:], \
                                                              img_gtruth_data[slice,:,:], \
                                                              img_mask_data[slice,:,:], \
                                                              is_extract_more_csf)
                
                # update database
                patches_training_imgs_2d_slice_temp  = np.append(patches_training_imgs_2d_slice_temp,imgs_patches, axis=0)
                patches_training_gtruth_2d_slice_temp  = np.append(patches_training_gtruth_2d_slice_temp,gt_patches, axis=0)
        
            patches_training_imgs_2d_temp  = np.append(patches_training_imgs_2d_temp,patches_training_imgs_2d_slice_temp, axis=0)
            patches_training_gtruth_2d_temp  = np.append(patches_training_gtruth_2d_temp,patches_training_gtruth_2d_slice_temp, axis=0)
               
        patches_training_imgs_2d  = np.append(patches_training_imgs_2d,patches_training_imgs_2d_temp, axis=0)
        patches_training_gtruth_2d  = np.append(patches_training_gtruth_2d,patches_training_gtruth_2d_temp, axis=0)
        j += 1
        X  = patches_training_imgs_2d.shape
        Y  = patches_training_gtruth_2d.shape
        print('shape im: [{0} , {1} , {2}]'.format(X[0], X[1], X[2])) 
        print('shape gt: [{0} , {1} , {2}, {3}]'.format(Y[0], Y[1], Y[2], Y[3]))

    #convert to single precission
    patches_training_imgs_2d = patches_training_imgs_2d.astype('float32')
    patches_training_imgs_2d = np.expand_dims(patches_training_imgs_2d, axis=3)
    
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
    
    X  = patches_training_imgs_2d.shape
    Y  = patches_training_gtruth_2d.shape
    
    print('-'*30)
    print('Training set detail...')
    print('-'*30)
    print('shape im: [{0} , {1} , {2}, {3}]'.format(X[0], X[1], X[2], X[3]))
    print('shape gt: [{0} , {1} , {2}, {3}]'.format(Y[0], Y[1], Y[2], Y[3]))

    S  = patches_training_imgs_2d.shape
    print('Done: {0} 2d patches added from {1} volume images'.format(S[0], j))
    print('Loading done.')

    print('Saving to .npy files done.')

    # save train or validation
    if is_train:
        np.save('imdbs_2d_patch/patches_training_imgs_2d.npy', patches_training_imgs_2d)
        np.save('imdbs_2d_patch/patches_training_gtruth_2d.npy', patches_training_gtruth_2d)
    else:
        np.save('imdbs_2d_patch/patches_val_imgs_2d.npy', patches_training_imgs_2d)
        np.save('imdbs_2d_patch/patches_val_gtruth_2d.npy', patches_training_gtruth_2d)        
    print('Saving to .npy files done.')


# extract 2d patches
def extract_2d_patches(img_data, gt_data, mask_data, is_extract_more_csf):
    patch_shape =(patch_size,patch_size)
    # empty matrix to hold patches
    imgs_patches_per_slice=np.empty(shape=[0,patch_size,patch_size], dtype='int16')
    gt_patches_per_slice=np.empty(shape=[0,patch_size,patch_size], dtype='int16')
    mask_patches_per_slice=np.empty(shape=[0,patch_size,patch_size], dtype='int16')
      
    img_patches = extract_patches(img_data, patch_shape, extraction_step)
    gt_patches = extract_patches(gt_data, patch_shape, extraction_step)
    mask_patches = extract_patches(mask_data, patch_shape, extraction_step)

    # extract patches which has center pixel lying inside mask    
    rows = []; cols = []
    for i in range(0,mask_patches.shape[0]):        
        for j in range(0,mask_patches.shape[1]):
            a1 = mask_patches.item((i,j,int(patch_size/2-1),int(patch_size/2-1)))
            a2 = mask_patches.item((i,j,int(patch_size/2-1),int(patch_size/2)))
            a3 = mask_patches.item((i,j,int(patch_size/2),int(patch_size/2-1)))
            a4 = mask_patches.item((i,j,int(patch_size/2),int(patch_size/2)))          
            Sum = a1 + a2 + a3 + a4
            if Sum > 0:
                rows.append(i)
                cols.append(j)
            
    # number of n0m zero patches
    N = len(rows)

    # select nonzeropatches index
    selected_img_patches = img_patches[rows,cols,:,:]
    selected_gt_patches  = gt_patches [rows,cols,:,:]
    
    # update database
    imgs_patches_per_slice  = np.append(imgs_patches_per_slice,selected_img_patches, axis=0)
    gt_patches_per_slice  = np.append(gt_patches_per_slice,selected_gt_patches, axis=0)
    
    #extract more pathes for CSF
    if is_extract_more_csf:
        #creat CSF mask
        extraction_step_csf  = extraction_step_csf_only
        img_patches_csf = extract_patches(img_data, patch_shape, extraction_step_csf)
        gt_patches_csf = extract_patches(gt_data, patch_shape, extraction_step_csf)
    
        # extract CSF patches with small step  
        rows = []; cols = []
        for i in range(0,gt_patches_csf.shape[0]):        
            for j in range(0,gt_patches_csf.shape[1]):
                a1 = gt_patches_csf.item((i,j,int(patch_size/2-1),int(patch_size/2-1)))
                a2 = gt_patches_csf.item((i,j,int(patch_size/2-1),int(patch_size/2)))
                a3 = gt_patches_csf.item((i,j,int(patch_size/2),int(patch_size/2-1)))
                a4 = gt_patches_csf.item((i,j,int(patch_size/2),int(patch_size/2)))
                Sum = (a1==1 or a2==1 or a3==1 or a4==1)
                if Sum:
                    rows.append(i)
                    cols.append(j)

        N = len(rows)
        if N is not 0:    
            csf_more_img_patches = img_patches_csf[rows,cols,:,:]
            csf_more_gt_patches = gt_patches_csf[rows,cols,:,:]
    
            # update database
            imgs_patches_per_slice  = np.append(imgs_patches_per_slice,csf_more_img_patches, axis=0)
            gt_patches_per_slice  = np.append(gt_patches_per_slice,csf_more_gt_patches, axis=0)
    
    # convert to categorical
    gt_patches_per_slice = separate_labels(gt_patches_per_slice)
    return imgs_patches_per_slice, gt_patches_per_slice


# separate labels
def separate_labels(patch_3d_volume):
    result =np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
    N = patch_3d_volume.shape[0]
    # for each class do:
    for V in range(N):
        V_patch = patch_3d_volume[V , :, :]
        U  = np.unique(V_patch)
        unique_values = list(U)
        result_v =np.empty(shape=[patch_size,patch_size,0], dtype='int16')
        if num_classes==3:
            start_point = 1
        else:
            start_point = 0
        for label in range(start_point,4):
            if label in unique_values:
                im_patch = V_patch == label
                im_patch = im_patch*1
            else:
                im_patch = np.zeros((V_patch.shape))
             
            im_patch = np.expand_dims(im_patch, axis=2) 
            result_v  = np.append(result_v,im_patch, axis=2)
        result_v = np.expand_dims(result_v, axis=0) 
        result  = np.append(result,result_v, axis=0)
    return result

# load train npy    
def load_train_data():
    x= np.load('imdbs_2d_patch/patches_training_imgs_2d.npy')
    y= np.load('imdbs_2d_patch/patches_training_gtruth_2d.npy')
    
    index=np.arange(x.shape[0])
    np.random.shuffle(index)
    val_num=int(len(index)*0.15)
    
    x_val=x[index[:val_num]]
    y_val=y[index[:val_num]]
    x_train=x[index[val_num:]]
    y_train=y[index[val_num:]]
    return x_train,y_train,x_val,y_val    


# main
if __name__ == '__main__':
    if 'imdbs_2d_patch' not in os.listdir(os.curdir):
        os.mkdir('imdbs_2d_patch')
    test_path = '../../datasets/testdata'
    train_imgs_path = '../../datasets/traindata'
    print(train_imgs_path)
    is_extract_more_csf = 0
    create_npy_data(train_imgs_path, is_extract_more_csf, 1)
