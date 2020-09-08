import pandas as pd
from number_extraction import *
# from open_file import *
from grid_ext import grid_ext
from lines_ext import *
from cells_ext import *
from skimage import color
import cv2
import numpy as np
import heapq

start_c = 2
end_c = 7
super_l = []
l = []
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    #blank_ch = 255*np.ones_like(label_hue)
    #labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    #labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    #0labeled_img[label_hue==0] = 0

    #cv2.imshow('labeled.png', labeled_img)
    #cv2.imshow('labeled.png', label_hue)
    #cv2.waitKey()
    return label_hue
    
    
    

# image = cv.imread('/home/badrivishal/Desktop/431.jpg')
image = cv.imread('sample1.jpg')

cv.imshow('original_img', cv.resize(image, (950, 850)))
cv.waitKey(0)

image = grid_ext(image)

temp=0
row1=image.shape[0]
col1=image.shape[1]
if col1>row1:
	temp=1
	for i in range(764,872):
		for j in range(62,145):
			if(image[j][i][0])<50:
				temp=2
if(temp==1):
	image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
elif(temp==2):
	image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

6
'''
row=image.shape[0]
col=image.shape[1]

for j in range(col):
    for i in range(row):
        if image[i][j][0] > 160:
        	image[i][j][0]=255
        	image[i][j][1]=255
        	image[i][j][2]=255
        else :
        	image[i][j][0]=0
        	image[i][j][1]=0
        	image[i][j][2]=0
'''

cv.imshow('hello', image)
cv.waitKey(0)

row=image.shape[0]
col=image.shape[1]
row_f=int(row/5.5)
#col_f=int(col/2.1)
#col_f=0
image = image[row_f:row,1:col]
cv2.imshow("cropped", image)
cv2.waitKey(0)

v_lines, h_lines = lines_ext(image, 225, 8)

ret_cell_fin = cells_ext(h_lines, v_lines, 0, 2, 10, 7)

cell_image = image.copy()

#model = load_model('digit_model.h5')
model = load_model('a2.h5')

super_l = []
l = []
str_num = ''

# print(len(ret_cell_fin))
# print(len(v_lines))
# print(len(h_lines))

# start_r = 2
start_c = 2
# end_r = 11
end_c = 7

b = np.zeros(256, dtype = int)
kernel1 = np.ones((5,5), np.uint8)
kernel2 = np.ones((3,3), np.uint8) 
#kernel1=(1,1)
#kernel2=(5,5)
#for m in range(len(ret_cell_fin)):
for m in range(len(ret_cell_fin)):
    p1, p2, p3, p4 = ret_cell_fin[m]

    cell = cell_image[p1[1]:p3[1], p1[0]:p2[0]]
    row=cell.shape[0]
    col=cell.shape[1]
    
    cell = cell[3:(row-3), 3:col-3]
    #cv2.imshow("cropped_cell", cell)
    row=cell.shape[0]
    col=cell.shape[1]
    cell_temp=cell.copy()
    #print("hello")
    img_ori = cv2.cvtColor(cell_temp, cv2.COLOR_BGR2GRAY)
    cv.imshow('img_ori', img_ori)
    cv.waitKey(0)
    threshold=0
    for i in range(row):
    	threshold=threshold+img_ori[i][2]
    for i in range(row):
    	threshold=threshold+img_ori[i][col-3]
    for i in range(col):
    	threshold=threshold+img_ori[2][i]
    for i in range(col):
    	threshold=threshold+img_ori[row-3][i]
    threshold=threshold/(2*(row+col))
    threshold=threshold-10
    img=img_ori.copy()
    for i in range(row):
    	for j in range(col):
    		if img[i][j]<threshold:
    			img[i][j]=255
    		else:
    			img[i][j]=0
    			
    #img = cv2.dilate(img, kernel2, iterations=1)
    #img = cv2.erode(img, kernel1, iterations=1)
    #img = cv2.dilate(img, kernel2, iterations=1) 	
    #cv.imshow('hello2', img)
    #cv.waitKey(0)
    #num_labels, labels_im = cv2.connectedComponents(img)
    
    img1=img.copy()
    for i in range(2,row-2):
        for j in range(2,col-2):
            if img1[i][j]==255:
                if img1[i-1][j]==0 and img1[i+1][j]==0:
                    img[i][j]=0
    
    
    ret, labels = cv2.connectedComponents(img)
    for i in range(256):
        b[i]=0
    label_hue=img.copy()
    for i in range(row):
    	for j in range(col):
    	    label_hue[i][j]=0
    row_num=0
    pos_im=[1000,1000,1000]
    val_im=[-1,-1,-1]
    loop_count=0
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        #val=int(255/label)
        mask[labels == label] = 255
        cv2.imshow('component',mask)
        cv2.waitKey(0)
        f_i=100000
        f_j=100000
        l_i=0
        l_j=0
        count=0
        for j in range(col):
            for i in range(row):
                if mask[i][j]==255:
                    #print(count)
                    count=count+1
                    #print("count1")
                    #print(loop_count) 
                    if loop_count<2 and pos_im[loop_count]==1000:
                    	pos_im[loop_count]=j
                    if l_i<i:
                        l_i=i
                    if l_j<j:
                        l_j=j
                    if f_i>i:
                        f_i=i
                    if f_j>j:
                        f_j=j
        if f_i < l_i and f_j < l_j and count > 30: 
            #print("count")
            #print(count)
            if f_i >0 :
                f_i=f_i-1
            if f_j >0 :
                f_j=f_j-1
            if l_i <(row-1) :
                l_i=l_i+1
            if l_j < (col-1) :
                l_j=l_j+1
            crop_img = mask[(f_i):(l_i), (f_j):(l_j)].copy()
            cv2.imshow("cropped", crop_img)
            cv2.waitKey()
            color = 0
            result = np.full((((l_i-f_i)+24),((l_j-f_j)+24)), color, dtype=np.uint8)
            result[12:((l_i-f_i)+12), 12:((l_j-f_j)+12)] = crop_img
            #result = np.full((((l_i-f_i)+24),((l_j-f_j)+30)), color, dtype=np.uint8)
            #result[12:((l_i-f_i)+12), 15:((l_j-f_j)+15)] = crop_img
            #result = np.full((((l_i-f_i)+24),((l_j-f_j)+28)), color, dtype=np.uint8)
            #result[12:((l_i-f_i)+12), 14:((l_j-f_j)+14)] = crop_img
            #result = np.full((((l_i-f_i)+24),((l_j-f_j)+26)), color, dtype=np.uint8)
            #result[12:((l_i-f_i)+12), 13:((l_j-f_j)+13)] = crop_img
            #result = np.full((((l_i-f_i)+24),((l_j-f_j)+32)), color, dtype=np.uint8)
            #result[12:((l_i-f_i)+12), 16:((l_j-f_j)+16)] = crop_img
            cv2.imshow("result", result)
            cv2.waitKey()
            #result = cv2.resize(result,  (300, 300)) 
            img1 = cv2.resize(result,  (28, 28)) 
            img1.reshape((28,28)).astype('float32')
            batch = np.expand_dims(img1,axis=0)
            #print(batch.shape) # (1, 28, 28)
            batch = np.expand_dims(batch,axis=3)
            batch=batch/255
            digit = model.predict_classes(batch)
            val_im[loop_count]=digit[0]
            loop_count+=1
            #print("hiiiiiiiiii")
            #print(digit[0])
        elif loop_count<2:
            pos_im[loop_count]=1000
    if val_im[0]==-1:
    	print(val_im[1])
    	l.append(val_im[1])
    elif val_im[1]==-1:
    	print(val_im[0])
    	l.append(val_im[0])
    elif pos_im[0]<pos_im[1]:
        num=((val_im[0]*10)+val_im[1])
        print(num)
        l.append(num)
    else:
        num=((val_im[1]*10)+val_im[0])
        print(num)
        l.append(num)
    #print(val_im)
    #0print(pos_im)
    print("hiiiiiiiiiii")
    if (m+1) % (end_c - start_c + 1) == 0:
        super_l.append(l)
        l = []   
    
original_df = pd.DataFrame(super_l)

df = original_df.copy()                                                            # change the variable names once done

print("hello")
print(df)	
