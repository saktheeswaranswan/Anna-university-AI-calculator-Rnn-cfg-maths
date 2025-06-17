https://stackoverflow.com/questions/50390836/understanding-darknets-yolo-cfg-config-files

onth ago
Modified 5 years ago
Viewed 56k times

Report this ad
49

I have searched around the internet but found very little information around this, I don't understand what each variable/value represents in yolo's .cfg files. So I was hoping some of you could help, I don't think I'm the only one having this problem, so if anyone knows 2 or 3 variables please post them so that people who needs such info in the future might find them.

The main one that I'd like to know are :

batch
subdivisions

decay

momentum

channels

filters

activation

yolodarknet
Share
Improve this question
Follow
asked May 17, 2018 at 11:55
Reda Drissi's user avatar
Reda Drissi
1,70244 gold badges2121 silver badges3434 bronze badges
Add a comment
4 Answers
Sorted by:

Highest score (default)
55

Here is my current understanding of some of the variables. Not necessarily correct though:

[net]
batch: That many images+labels are used in the forward pass to compute a gradient and update the weights via backpropagation.
subdivisions: The batch is subdivided in this many "blocks". The images of a block are ran in parallel on the gpu.
decay: Maybe a term to diminish the weights to avoid having large values. For stability reasons I guess.
channels: Better explained in this image :
On the left we have a single channel with 4x4 pixels, The reorganization layer reduces the size to half then creates 4 channels with adjacent pixels in different channels. figure

momentum: I guess the new gradient is computed by momentum * previous_gradient + (1-momentum) * gradient_of_current_batch. Makes the gradient more stable.
adam: Uses the adam optimizer? Doesn't work for me though
burn_in: For the first x batches, slowly increase the learning rate until its final value (your learning_rate parameter value). Use this to decide on a learning rate by monitoring until what value the loss decreases (before it starts to diverge).
policy=steps: Use the steps and scales parameters below to adjust the learning rate during training
steps=500,1000: Adjust the learning rate after 500 and 1000 batches
scales=0.1,0.2: After 500, multiply the LR by 0.1, then after 1000 multiply again by 0.2
angle: augment image by rotation up to this angle (in degree)
layers
filters: How many convolutional kernels there are in a layer.
activation: Activation function, relu, leaky relu, etc. See src/activations.h
stopbackward: Do backpropagation until this layer only. Put it in the panultimate convolution layer before the first yolo layer to train only the layers behind that, e.g. when using pretrained weights.
random: Put in the yolo layers. If set to 1 do data augmentation by resizing the images to different sizes every few batches. Use to generalize over object sizes.
Many things are more or less self-explanatory (size, stride, batch_normalize, max_batches, width, height). If you have more questions, feel free to comment.

Again, please keep in mind that I am not 100% certain about many of those.

Share
Improve this answer
Follow
edited Jun 5, 2018 at 14:05
answered Jun 5, 2018 at 9:40
FelEnd's user avatar
FelEnd
88899 silver badges1212 bronze badges
2
About the channels: yes, I cannot find a connection between the image channels and the cfg-parameter channels in the source. darknet seems to be hardcoded to color images. I'll edit my answer accordingly – 
FelEnd
 CommentedJun 5, 2018 at 10:54
2
I am unsure about your explanation of channels. When talking about the input to the network (the parameter is in the [network] section) people seem to use "channel" to refer to the color channels. For later layers "channels" and "depth" seems to be interchangable. In the yolo cfg, the number of (output) channels of a layer is given by "filters" (as each filter produces one channel). I don't see how your edit explains what the parameter actually does. – 
FelEnd
 CommentedJun 5, 2018 at 13:07 
3
How min_crop/max_crop work? I.e. min_crop can be > image dimensions (as shown in cifar training example) – 
apatsekin
 CommentedOct 9, 2018 at 23:27
5
What about anchors? – 
Jürgen K.
 CommentedApr 15, 2019 at 14:33
4
What about mask – 
Jürgen K.
 CommentedApr 15, 2019 at 14:33
Show 5 more comments

Report this ad
17

More complete explanation about the cfg parameters, copied from the author of YOLO v4 https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section and https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers

Below is only the snapshot of the documentation, please refer to the above links for a better format

CFG-Parameters in the [net] section:

[net] section
batch=1 - number of samples (images, letters, ...) which will be precossed in one batch

subdivisions=1 - number of mini_batches in one batch, size mini_batch = batch/subdivisions, so GPU processes mini_batch samples at once, and the weights will be updated for batch samples (1 iteration processes batch images)

width=416 - network size (width), so every image will be resized to the network size during Training and Detection

height=416 - network size (height), so every image will be resized to the network size during Training and Detection

channels=3 - network size (channels), so every image will be converted to this number of channels during Training and Detection

inputs=256 - network size (inputs) is used for non-image data: letters, prices, any custom data

max_chart_loss=20 - max value of Loss in the image chart.png

For training only

Contrastive loss:

contrastive=1 - use Supervised contrastive loss for training Classifier (should be used with [contrastive] layer)

unsupervised=1 - use Unsupervised contrastive loss for training Classifier on images without labels (should be used with contrastive=1 parameter and with [contrastive] layer)

Data augmentation:

angle=0 - randomly rotates images during training (classification only)

saturation = 1.5 - randomly changes saturation of images during training

exposure = 1.5 - randomly changes exposure (brightness) during training

hue=.1 - randomly changes hue (color) during training https://en.wikipedia.org/wiki/HSL_and_HSV

blur=1 - blur will be applied randomly in 50% of the time: if 1 - will be blured background except objects with blur_kernel=31, if >1 - will be blured whole image with blur_kernel=blur (only for detection and if OpenCV is used)

min_crop=224 - minimum size of randomly cropped image (classification only)

max_crop=448 - maximum size of randomly cropped image (classification only)

aspect=.75 - aspect ration can be changed during croping from 0.75 - to 1/0.75 (classification only)

letter_box=1 - keeps aspect ratio of loaded images during training (detection training only, but to use it during detection-inference - use flag -letter_box at the end of detection command)

cutmix=1 - use CutMix data augmentation (for Classifier only, not for Detector)

mosaic=1 - use Mosaic data augmentation (4 images in one)

mosaic_bound=1 - limits the size of objects when mosaic=1 is used (does not allow bounding boxes to leave the borders of their images when Mosaic-data-augmentation is used)

data augmentation in the last [yolo]-layer

jitter=0.3 - randomly changes size of image and its aspect ratio from x(1 - 2*jitter) to x(1 + 2*jitter)
random=1 - randomly resizes network size after each 10 batches (iterations) from /1.4 to x1.4 with keeping initial aspect ratio of network size
adversarial_lr=1.0 - Changes all detected objects to make it unlike themselves from neural network point of view. The neural network do an adversarial attack on itself

attention=1 - shows points of attention during training

gaussian_noise=1 - add gaussian noise

Optimizator:

momentum=0.9 - accumulation of movement, how much the history affects the further change of weights (optimizer)

decay=0.0005 - a weaker updating of the weights for typical features, it eliminates dysbalance in dataset (optimizer) http://cs231n.github.io/neural-networks-3/

learning_rate=0.001 - initial learning rate for training

burn_in=1000 - initial burn_in will be processed for the first 1000 iterations, current_learning rate = learning_rate * pow(iterations / burn_in, power) = 0.001 * pow(iterations/1000, 4) where is power=4 by default

max_batches = 500200 - the training will be processed for this number of iterations (batches)

policy=steps - policy for changing learning rate: constant (by default), sgdr, steps, step, sig, exp, poly, random (f.e., if policy=random - then current learning rate will be changed in this way = learning_rate * pow(rand_uniform(0,1), power))

power=4 - if policy=poly - the learning rate will be = learning_rate * pow(1 - current_iteration / max_batches, power)

sgdr_cycle=1000 - if policy=sgdr - the initial number of iterations in cosine-cycle

sgdr_mult=2 - if policy=sgdr - multiplier for cosine-cycle https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163

steps=8000,9000,12000 - if policy=steps - at these numbers of iterations the learning rate will be multiplied by scales factor

scales=.1,.1,.1 - if policy=steps - f.e. if steps=8000,9000,12000, scales=.1,.1,.1 and the current iteration number is 10000 then current_learning_rate = learning_rate * scales[0] * scales[1] = 0.001 * 0.1 * 0.1 = 0.00001

label_smooth_eps=0.1 - use label smoothing for training Classifier

For training Recurrent networks:

Object Detection/Tracking on Video - if [conv-lstm] or [crnn] layers are used in additional to [connected] and [convolutional] layers

Text generation - if [lstm] or [rnn] layers are used in additional to [connected] layers

track=1 - if is set 1 then the training will be performed in Recurrents-tyle for image sequences

time_steps=16 - training will be performed for a random image sequence that contains 16 images from train.txt file

for [convolutional]-layers: mini_batch = time_steps*batch/subdivisions
for [conv_lstm]-recurrent-layers: mini_batch = batch/subdivisions and sequence=16
augment_speed=3 - if set 3 then can be used each 1st, 2nd or 3rd image randomly, i.e. can be used 16 images with indexes 0, 1, 2, ... 15 or 110, 113, 116, ... 155 from train.txt file

sequential_subdivisions=8 - lower value increases the sequence of images, so if time_steps=16 batch=16 sequential_subdivisions=8, then will be loaded time_steps*batch/sequential_subdivisions = 16*16/8 = 32 sequential images with the same data-augmentation, so the model will be trained for sequence of 32 video-frames

seq_scales=0.5, 0.5 - increasing sequence of images at some steps, i.e. the coefficients to which the original sequential_subdivisions value will be multiplied (and batch will be dividied, so the weights will be updated rarely) at correspond steps if is used policy=steps or policy=sgdr

CFG-Parameters in the different layers
Image processing [N x C x H x W]:

[convolutional] - convolutional layer

batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)

filters=64 - number of kernel-filters (1 by default)

size=3 - kernel_size of filter (1 by default)

groups = 32 - number of groups for grouped-convolutional (depth-wise) (1 by default)

stride=1 - stride (offset step) of kernel filter (1 by default)

padding=1 - size of padding (0 by default)

pad=1 - if 1 will be used padding = size/2, if 0 the will be used parameter padding= (0 by default)

dilation=1 - size of dilation (1 by default)

activation=leaky - activation function after convolution: logistic (by default), loggy, relu, elu, selu, relie, plse, hardtan, lhtan, linear, ramp, leaky, tanh, stair, relu6, swish, mish

[activation] - separate activation layer

activation=leaky - activation function: linear (by default), loggy, relu, elu, selu, relie, plse, hardtan, lhtan, linear, ramp, leaky, tanh, stair
[batchnorm] - separate Batch-normalization layer
[maxpool] - max-pooling layer (the maximum value)

size=2 - size of max-pooling kernel

stride=2 - stirde (offset step) of max-pooling kernel

[avgpool] - average pooling layer input W x H x C -> output 1 x 1 x C
[shortcut] - residual connection (ResNet)

from=-3,-5 - relative layer numbers, preforms element-wise adding of several layers: previous-layer and layers specified in from= parameter

weights_type=per_feature - will be used weights for shortcut y[i] = w1*layer1[i] + w2*layer2[i] ...

per_feature - 1 weights per layer/feature
per_channel - 1 weights per channel
none - weights will not be used (by default)
weights_normalization=softmax - will be used weights normalization

softmax - softmax normalization
relu - relu normalization
none - without weights normalization - unbound weights (by default)
activation=linear - activation function after shortcut/residual connection (linear by default)

[upsample] - upsample layer (increase W x H resolution of input by duplicating elements)

stride=2 - factor for increasing both Width and Height (new_w = w*stride, new_h = h*stride)
[scale_channels] - scales channels (SE: squeeze-and-excitation blocks) or (ASFF: adaptively spatial feature fusion) -it multiplies elements of one layer by elements of another layer

from=-3 - relative layer number, performs multiplication of all elements of channel N from layer -3, by one element of channel N from the previous layer -1 (i.e. for(int i=0; i < b*c*h*w; ++i) output[i] = from_layer[i] * previous_layer[i/(w*h)]; )

scale_wh=0 - SE-layer (previous layer 1x1xC), scale_wh=1 - ASFF-layer (previous layer WxHx1)

activation=linear - activation function after scale_channels-layer (linear by default)

[sam] - Spatial Attention Module (SAM) - it multiplies elements of one layer by elements of another layer

from=-3 - relative layer number (this and previous layers should be the same size WxHxC)
[reorg3d] - reorg layer (resize W x H x C)

stride=2 - if reverse=0 input will be resized to W/2 x H/2 x C4, if reverse=1thenW2 x H*2 x C/4`, (1 by default)

reverse=1 - if 0(by default) then decrease WxH, if1thenincrease WxH (0 by default)

[reorg] - OLD reorg layer from Yolo v2 - has incorrect logic (resize W x H x C) - depracated

stride=2 - if reverse=0 input will be resized to W/2 x H/2 x C4, if reverse=1thenW2 x H*2 x C/4`, (1 by default)

reverse=1 - if 0(by default) then decrease WxH, if1thenincrease WxH (0 by default)

[route] - concatenation layer, Concat for several input-layers, or Identity for one input-layer

layers = -1, 61 - layers that will be concatenated, output: W x H x C_layer_1 + C_layer_2
if index < 0, then it is relative layer number (-1 means previous layer)
if index >= 0, then it is absolute layer number
[yolo] - detection layer for Yolo v3 / v4

mask = 3,4,5 - indexes of anchors which are used in this [yolo]-layer

anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326 - initial sizes if bounded_boxes that will be adjusted

num=9 - total number of anchors

classes=80 - number of classes of objects which can be detected

ignore_thresh = .7 - keeps duplicated detections if IoU(detect, truth) > ignore_thresh, which will be fused during NMS (is used for training only)

truth_thresh = 1 - adjusts duplicated detections if IoU(detect, truth) > truth_thresh, which will be fused during NMS (is used for training only)

jitter=.3 - randomly crops and resizes images with changing aspect ratio from x(1 - 2*jitter) to x(1 + 2*jitter) (data augmentation parameter is used only from the last layer)

random=1 - randomly resizes network for each 10 iterations from 1/1.4 to 1.4(data augmentation parameter is used only from the last layer)

resize=1.5 - randomly resizes image in range: 1/1.5 - 1.5x

max=200 - maximum number of objects per image during training

counters_per_class=100,10,1000 - number of objects per class in Training dataset to eliminate the imbalance

label_smooth_eps=0.1 - label smoothing

scale_x_y=1.05 - eliminate grid sensitivity

iou_thresh=0.2 - use many anchors per object if IoU(Obj, Anchor) > 0.2

iou_loss=mse - IoU-loss: mse, giou, diou, ciou

iou_normalizer=0.07 - normalizer for delta-IoU

cls_normalizer=1.0 - normalizer for delta-Objectness

max_delta=5 - limits delta for each entry

[crnn] - convolutional RNN-layer (recurrent)

batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)

size=1 - convolutional kernel_size of filter (1 by default)

pad=0 - if 1 will be used padding = size/2, if 0 the will be used parameter padding= (0 by default)

output = 1024 - number of kernel-filters in one output convolutional layer (1 by default)

hidden=1024 - number of kernel-filters in two (input and hidden) convolutional layers (1 by default)

activation=leaky - activation function for each of 3 convolutional-layers in the [crnn]-layer (logistic by default)

[conv_lstm] - convolutional LSTM-layer (recurrent)

batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)

size=3 - convolutional kernel_size of filter (1 by default)

padding=1 - convolutional size of padding (0 by default)

pad=1 - if 1 will be used padding = size/2, if 0 the will be used parameter padding= (by default)

stride=1 - convolutional stride (offset step) of kernel filter (1 by default)

dilation=1 - convolutional size of dilation (1 by default)

output=256 - number of kernel-filters in each of 8 or 11 convolutional layers (1 by default)

groups=4 - number of groups for grouped-convolutional (depth-wise) (1 by default)

state_constrain=512 - constrains LSTM-state values [-512; +512] after each inference (time_steps*32 by default)

peephole=0 - if 1 then will be used Peephole (additional 3 conv-layers), if 0 will not (1 by default)

bottleneck=0 - if 1 then will be used reduced optimal versionn of conv-lstm layer

activation=leaky - activation function for each of 8 or 11 convolutional-layers in the [conv_lstm]-layer (linear by default)

lstm_activation=tanh - activation for G (gate: g = tanh(wg + ug)) and C (memory cell: h = o * tanh(c))

Detailed-architecture-of-the-peephole-LSTM

Free-form data processing [Inputs]:

[connected] - fully connected layer
output=256 - number of outputs (1 by default), so number of connections is equal to inputs*outputs
activation=leaky - activation after layer (logistic by default)
[dropout] - dropout layer
probability=0.5 - dropout probability - what part of inputs will be zeroed (0.5 = 50% by default)

dropblock=1 - use as DropBlock

dropblock_size_abs=7 - size of DropBlock in pixels 7x7

[softmax] - SoftMax CE (cross entropy) layer - Categorical cross-entropy for multi-class classification
[contrastive] - Contrastive loss layer for Supervised and Unsupervised learning (should be set [net] contrastive=1 and optionally [net] unsupervised=1)

classes=1000 - number of classes

temperature=1.0 - temperature

[cost] - cost layer calculates (linear)Delta and (squared)Loss
type=sse - cost type: sse (L2), masked, smooth (smooth-L1) (SSE by default)
[rnn] - fully connected RNN-layer (recurrent)
batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)
output = 1024 - number of outputs in one connected layer (1 by default)
hidden=1024 - number of outputs in two (input and hidden) connected layers (1 by default)
activation=leaky - activation after layer (logistic by default)
[lstm] - fully connected LSTM-layer (recurrent)
batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)
output = 1024 - number of outputs in all connected layers (1 by default)
[gru] - fully connected GRU-layer (recurrent)
batch_normalize=1 - if 1 - will be used batch-normalization, if 0 will not (0 by default)
output = 1024 - number of outputs in all 

connected layers (1 by default)
