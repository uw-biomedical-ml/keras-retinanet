import cv2
import os
import sys
import csv

load_size_h, load_size_w = 1024, 1024
patch_size_h, patch_size_w = 512, 512 ##256, 256
basedir = '/data/VLDLR/data'
TARGET_NAME = 'T'
EMPTY = [''] * 5

# code from https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d
def findBbox(im):
  imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,127,255,0)
  rlt = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #print('rlt len', len(rlt))
  #print(rlt)
  _, contours, hierarchy = rlt
  # with each contour, draw boundingRect in green
  # a minAreaRect in red and
  # a minEnclosingCircle in blue
  bbox = []
  for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    bbox.append([x, y, x+w, y+h, TARGET_NAME])
    #print(x, y, w, h)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

  return im, bbox

def get_desired_size(h, patch_size):
  nh, rh = int(h / patch_size), h % patch_size
  desire_h = nh * patch_size
  if rh > 0:
    desire_h += patch_size
  return desire_h

def makeBorders(desire_h, desire_w, h, w, img, label):
  top = int((desire_h - h) / 2)
  bottom = desire_h - top - h
  left = int((desire_w - w) / 2)
  right = desire_w - w- left
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
  label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
  return img, label

def get_patches(imgfile, labelfile, outputdir, sanityChkDir, csvfilepath, idx, h=load_size_h, w=load_size_w):
  img = cv2.imread(imgfile)
  ih, iw, c = img.shape
  if h is not None:
    img = cv2.resize(img, (w, h))
  label = cv2.imread(labelfile)
  lh, lw, lc = label.shape
  if h is not None:
    label = cv2.resize(label, (w, h))
  #print(h, w, lh, lw)
  #print(img.shape, label.shape)
  desire_h = get_desired_size(h, patch_size_h)
  desire_w = get_desired_size(w, patch_size_w)
  if desire_h != h or desire_w != w:
    img, label = makeBorders(desire_h, desire_w, h, w, img, label)
  #print(img.shape, label.shape)
  #sys.exit()
  
  #basename = os.path.splitext(os.path.basename(imgfile))[0] ## bug! basename may not be unique!!
  img_bbox = []
  for i in range(int(desire_h / patch_size_h)):
    #if i != 0:
    #  continue
    for j in range(int(desire_w / patch_size_h)):
      #if j != 0:
      #  continue
      #print(i, j)
      imgname = '%s/%d_%d_%d.png' % (outputdir, idx, i, j)
      impatch = img[patch_size_h * (i) : patch_size_h * (i+1), patch_size_w * j : patch_size_w * (j+1)]
      cv2.imwrite(imgname, impatch)
      labelpatch = label[patch_size_h * (i) : patch_size_h * (i+1), patch_size_w * j : patch_size_w * (j+1)]
      labelname = '%s/%d_%d_%d_label.png' % (sanityChkDir, idx, i, j)
      cv2.imwrite(labelname, labelpatch)
      bboxpatch, bbox = findBbox(labelpatch)
      #print(impatch.shape, bboxpatch.shape)
      mergepatch = cv2.add(impatch, bboxpatch)
      mergename = '%s/%d_%d_%d.png' % (sanityChkDir, idx, i, j)
      cv2.imwrite(mergename, mergepatch)
      if len(bbox) > 0:
        img_bbox.append([imgname, bbox])
      else:
        img_bbox.append([imgname, [EMPTY]])

  with open(csvfilepath, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for imgname, bbox_arr in img_bbox:
      for bbox in bbox_arr:
        row = [imgname]
        row.extend(bbox)
        writer.writerow(row)

def process(split):
  outputdir = '%s/patch%d_%d/%s' % (basedir, patch_size_h, patch_size_w, split)
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)
  sanitydir = '%s/patch%d_%d/sanity/%s' % (basedir, patch_size_h, patch_size_w, split)
  if not os.path.exists(sanitydir):
    os.makedirs(sanitydir)

  A_filepath = '{}/{}_A.txt'.format(basedir, split)
  B_filepath = '{}/{}_B.txt'.format(basedir, split)
  with open(A_filepath) as f:
    content = f.readlines()
  A_files = [x.strip() for x in content]
  with open(B_filepath) as f:
      content = f.readlines()
  B_files = [x.strip() for x in content]
  csvfilepath = '%s/csv_data_file.csv' % outputdir
  for i, A_file in enumerate(A_files):
    #if i == 1:
    #  break
    print(split, i, A_file)
    B_file = B_files[i]
    get_patches(A_file, B_file, outputdir, sanitydir, csvfilepath, i)
  #imgfile = '/data/VLDLR/code/keras-retinanet/images/01.tif'
  #labelfile = '/data/VLDLR/code/keras-retinanet/images/01_label.jpg'
  #get_patches(imgfile, labelfile, outputdir, sanitydir, csvfilepath)

def createCSVClassFile(filepath):
  with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow([TARGET_NAME,0])

#createCSVClassFile('%s/csv_class_file.csv' % basedir)
process('train')
#process('val')
process('test')

def testFindBbox():
  labelfile = 'raw_label.jpg'
  label = cv2.imread(labelfile)
  print(label.shape)
  findBbox(label)    

#testFindBbox()

def merge(im1path, im2path):
  im1 = cv2.imread(im1path)
  im1 = cv2.resize(im1, (512, 512))
  im2 = cv2.imread(im2path)
  im2 = cv2.resize(im2,(512, 512))
  print(im1.shape, im2.shape)
  #im = cv2.merge((im2[:,:,0], im1[:,:,0], im1[:,:,2]))
  im = cv2.addWeighted(im1, im2)
  cv2.imwrite('01_merge.png', im)

#imgfile = '/data/VLDLR/code/keras-retinanet/images/01.tif'
#labelfile = '/data/VLDLR/code/keras-retinanet/images/01_label.jpg'
#merge(imgfile, labelfile)
