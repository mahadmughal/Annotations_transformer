from bs4 import BeautifulSoup
from skimage.util import random_noise
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import os
import cv2
import pdb

SOURCE_PATH = '/home/mahad/Desktop/my_work/labels_tranformation/tt2'
DESTINATION_PATH = '/home/mahad/Desktop/my_work/labels_tranformation/destination_folder'


def tag_data(tag, bs_data):

  return bs_data.find(tag).text


def add_noise_to_image(image, prob):
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]

  return output


def blur_image(img, kernel_size):
  img_blur = cv2.blur(img, (kernel_size, kernel_size))

  return blur_image


def grey_scale_image(img, percent):
  (row, col) = img.shape[0:2]

  for i in range(row):
    for j in range(col):
      img[i, j] = sum(img[i, j]) * percent

  return img


def resize_image(img_path, target_size=416):
  img = cv2.imread(xml_file_path[:len(xml_file_path)-3]+'png')  
  resized_img = cv2.resize(img, (target_size, target_size))

  return resized_img


def write_yolo_annotation(x_min, y_min, x_max, y_max, file_name):
  if os.path.isfile(os.path.join(DESTINATION_PATH, file_name) + '.txt'):
    return
  else:
    with open(os.path.join(DESTINATION_PATH, file_name) + ".txt", 'w') as f:
      f.write(' '.join(['0',
        str(round((x_max+x_min)/2,6)),
        str(round((y_max+y_min)/2,6)),
        str(round(x_max-x_min,6)),
        str(round(y_max-y_min,6))])+'\n')


def write_preprocessed_image(file_name, img):
  cv2.imwrite(os.path.join(DESTINATION_PATH , file_name+'.png'), img)


def main():

  parser = argparse.ArgumentParser(description="Preprocess the image")
  parser.add_argument('-b', type=int, default=10, help='defines the blur percentage')
  parser.add_argument('-n', type=int, default=5, help='defines the noise percentage')
  parser.add_argument('-g', type=int, default=25, help='defines the grayscling percentage')
  parser.add_argument('-s', type=int, default=416, help='defines desired size of an image' )

  arguments = parser.parse_args()

  for xml_file_path in tqdm(glob(os.path.join(SOURCE_PATH, '*.xml'))):
    with open(xml_file_path, 'r') as f:
      data = f.read()

    img = cv2.imread(xml_file_path[:len(xml_file_path)-3]+'png')
    img = cv2.resize(img, (arguments.s, arguments.s))

    bs_data = BeautifulSoup(data, 'xml')

    file_name = tag_data('filename', bs_data)

    img_width = int(tag_data('width', bs_data))
    img_height = int(tag_data('height', bs_data))

    x_min = int(tag_data('xmin', bs_data))/img_width
    y_min = int(tag_data('ymin', bs_data))/img_height
    x_max = int(tag_data('xmax', bs_data))/img_width
    y_max = int(tag_data('ymax', bs_data))/img_height

    write_preprocessed_image(file_name, img)
    write_yolo_annotation(x_min, y_min, x_max, y_max, file_name)

    # add noise to an image
    if arguments.n in range(5, 26):
      print(f'Adding noise to an image {file_name} upto {arguments.n} percent')
      noise_img = add_noise_to_image(img, arguments.n/100)

      write_preprocessed_image(f'noisy_{file_name}', img)
      write_yolo_annotation(x_min, y_min, x_max, y_max, f'noisy_{file_name}')
    else:
      print(f'Valid percentage of adding noise to an image is 5 to 25')
      print('So the image could not be made noisy due to invalid percentage')

    # make an image blur
    if arguments.b in range(10, 26):
      print(f'Blurring an image {file_name} upto {arguments.b} px')
      blur_img = blur_image(img, arguments.b)

      write_preprocessed_image(f'blur_{file_name}', img)
      write_yolo_annotation(x_min, y_min, x_max, y_max, f'blur_{file_name}')
    else:
      print(f'Valid percentage to blur an image is 10 to 25')
      print('So the image could not be made blur due to invalid kernel size')
    
    # grey-scaling an image
    if arguments.b in range(10, 36):
      print(f'Grey-scaling an image {file_name} upto {arguments.g} percent')
      blur_img = grey_scale_image(img, arguments.g/100)

      write_preprocessed_image(f'greyscaled_{file_name}', img)
      write_yolo_annotation(x_min, y_min, x_max, y_max, f'greyscaled_{file_name}')
    else:
      print(f'Valid percentage to grey-scale an image is 10 to 35')
      print('So the image could not be made grey-scaled due to invalid percentage')

    

main()
