# "Neural Art" in tensorflow

An implementation of ["A neural algorithm of Artistic style"](http://arxiv.org/abs/1508.06576), demonstrating the use of various Caffe cnn models (VGG and illustration2vec) in tensorflow.

<img src="vgg_result.png">

(VGG, default settings, 70 iterations)

<img src="i2v_result.png">

(illustration2vec, width=500, beta=10000, 100 iterations)

##Usage

### Step 0: Prepare the Caffe model
First, download either the VGG model or the illustration2vec model (\*.caffemodel), along with the prototxt (\*.prototxt):

- VGG: [https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)
- illustration2vec: [http://illustration2vec.net/](http://illustration2vec.net/)   (pre-trained model for tag prediction, version 2.0)

Then, convert the model to a binary format recognizable to tensorflow:

	python ./kaffe/kaffe.py [path.prototxt] [path.caffemodel] [output-path]

Note that Caffe is *not* required for conversion.

The converter included in this repo (all code inside ./kaffe) is a modified version of (an old version of) https://github.com/ethereon/caffe-tensorflow . The converter is modified to be capable of handling the illusration2vec neural network. Since the newer version of the converter requires preprocessing with the Caffe framework for old-format Caffe models (at the time of writing), I have included the converter which is based on the older code, which is capable of handling old-format Caffe models. 

### Step 1: Neural Art

	python neural-art-tf.py

Running `neural-art-tf.py` without options yields the default settings and input images. Available options are:

- `-m, --model`:      Model type - Use `vgg` or `i2v`
- `-mp, --modelpath`: Model file path - The path to the converted Caffe model in Step 0
- `-c, --content`:    Content image path
- `-s, --style`:      Style image path
- `-w, --width`:      Output image width
- `-i, --iters`:      Number of iterations
- `-a, --alpha`:      alpha (content weight)
- `-b, --beta`:       beta (style weight)

For example:


	python neural-art-tf.py -m vgg -mp ./vgg -c ./images/sd.jpg -s ./images/style.jpg -w 800

You can view the progress on tensorboard by running

	tensorboard --logdir=/tmp/na-logs

## References
- L. A. Gatys, et al., A neural algorithm of Artistic style, 2015, [http://arxiv.org/abs/1508.06576](http://arxiv.org/abs/1508.06576)
- [https://github.com/ethereon/caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) : The Caffe model converter
- [https://github.com/mattya/chainer-gogh](https://github.com/mattya/chainer-gogh) : Implementation in Chainer. Referenced the argument parser
- [https://github.com/anishathalye/neural-style](https://github.com/anishathalye/neural-style) : Another implementation in tensorflow. Referenced the learning rates


- `./kaffe/caffepb.py` was referenced from [https://github.com/ethereon/caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).
- `./kaffe/kaffe.py`, `./network.py`, `./models.py` are modified versions originally from [https://github.com/ethereon/caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow). (`./models.py` was originally `vgg.py`)
