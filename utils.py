import numpy as np
import scipy.misc
import os
from datetime import datetime as dt
import argparse
from models import VGG16, I2V

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
    for i in range(3):
        img[0,:,:,i] += mean[i]
    return img

def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

def read_image(path, w=None):
    img = scipy.misc.imread(path)
    # Resize if ratio is specified
    if w:
        r = w / np.float32(img.shape[1])
        img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
    img = img.astype(np.float32)
    img = img[None, ...]
    # Subtract the image mean
    img = sub_mean(img)
    return img

def save_image(im, iteration, out_dir):
    img = im.copy()
    # Add the image mean
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    scipy.misc.imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)
   
def parseArgs():
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--model', '-m', default='vgg',
                        help='Model type (vgg, i2v)')
    parser.add_argument('--modelpath', '-mp', default='vgg',
                        help='Model file path')
    parser.add_argument('--content', '-c', default='images/sd.jpg',
                        help='Content image path')
    parser.add_argument('--style', '-s', default='images/style.jpg',
                        help='Style image path')
    parser.add_argument('--width', '-w', default=800, type=int,
                        help='Output image width')
    parser.add_argument('--iters', '-i', default=5000, type=int,
                        help='Number of iterations')
    parser.add_argument('--alpha', '-a', default=1.0, type=float,
                        help='alpha (content weight)')
    parser.add_argument('--beta', '-b', default=200.0, type=float,
                        help='beta (style weight)')
    parser.add_argument('--device', default="/cpu:0")
    parser.add_argument('--out_dir', default="output")
    args = parser.parse_args()
    return args.content, args.style, args.modelpath, args.model, args.width, args.alpha, args.beta, args.iters, args.device, args

def getModel(image, params_path, model):
    if model == 'vgg':
        return VGG16(image, params_path)
    elif model == 'i2v':
        return I2V(image, params_path)
    else:
        print 'Invalid model name: use `vgg` or `i2v`'
        return None
