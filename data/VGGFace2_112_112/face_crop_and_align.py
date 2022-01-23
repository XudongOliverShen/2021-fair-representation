import cv2
import numpy as np
from skimage import transform as trans
import numpy as np
from tqdm import tqdm
from os.path import join as opj
import os
import pandas
import multiprocessing as mp

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_image(img_path, mode='rgb', layout='HWC'):
  if mode=='gray':
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  else:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if mode=='rgb':
      img = img[...,::-1]
    if layout=='CHW':
      img = np.transpose(img, (2,0,1))
  return img

class crop_and_align(object):
  def __init__(self):
    self.img_size = [112, 112]
    self.src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    self.src[:,0] += 8.0
    self.tform = trans.SimilarityTransform()
    
  def do(self, img, landmark):
    dst = landmark.astype(np.float32)
    self.tform.estimate(dst, self.src)
    M = self.tform.params[0:2,:]

    warped = cv2.warpAffine(img, M, (self.img_size[1],self.img_size[0]), borderValue = 0.0)

    return warped


if __name__ == '__main__':

  for task in ['test', 'train']:
    print(task)
    meta_file = 'dataset/identity_meta.csv'
    bb_file = 'dataset/bb_landmark/loose_bb_'+task+'.csv'
    lm_file = 'dataset/bb_landmark/loose_landmark_'+task+'.csv'
    in_dir = 'dataset/' + task
    out_dir = task + '/'
    num_parallel_threads = 8

    bb = pandas.read_csv(bb_file)
    lm = pandas.read_csv(lm_file)
    num_imgs = bb.shape[0]

    print('creating dirs storing new dataset...')
    # meta information is only used for creating folders
    meta = pandas.read_csv(meta_file, sep='\s*,\s*')
    if task is 'train':
      meta = meta[meta.Flag==1]
    elif task is 'test':
      meta = meta[meta.Flag==0]
    num_pp = meta.shape[0]

    for i in tqdm(range(num_pp)):
      pp = meta.iloc[i].Class_ID
      dir_path = opj(out_dir, pp)
      if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    print('dirs created!')

    def single_process(i):
      pp, idx = bb.iloc[i]['NAME_ID'].split('/')
      lmark = list(lm.iloc[i])[1:]
      lmark_1 = np.zeros([5,2])
      for i in range(5):
        lmark_1[i, 0] = lmark[i*2] 
        lmark_1[i, 1] = lmark[i*2 + 1] 
      lmark = lmark_1

      img_path = opj(in_dir, pp, idx + '.jpg')
      img = read_image(img_path, mode='rgb')

      warped = CA.do(img, lmark)
      bgr = warped[...,::-1]

      out_path = opj(out_dir, pp, idx + '.jpg')
      cv2.imwrite(out_path, bgr)

      return 1
    
    CA = crop_and_align()

    pool = mp.Pool(processes=num_parallel_threads)
    for _ in tqdm(pool.imap_unordered(single_process, range(num_imgs)), total=num_imgs):
      pass