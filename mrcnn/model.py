"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES]) # P2, P3, P4, P5, P6 对应原图的下采样倍率是 4，8，16，32，64 . 知道了下采样倍率和原图大小，就可以知道特征图的大小 [fan]


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 2倍下采样 [fan]
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True): # 这个函数在文档的"构建ResNet"中详细说明了 [fan]
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image) # 对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小。元组里的整数代表填充0的数目，两边都会补零 [fan] 卷积的操作中,如果使用same,或valid这种模式,有时候会不灵活。必要的时候,需要我们自己去进行补零操作,庆幸的是keras的补零操作是非常灵活的。https://blog.csdn.net/lujiandong1/article/details/54918320
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x) # 64是卷积核的个数，(7,7)是卷积核的长宽 [fan]
    x = BatchNorm(name='bn_conv1')(x, training=train_bn) # 批量归一化层 [fan]
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn) # 注意这里的步长为1，与论文里的架构描述吻合 。下面的Stage的步长默认是2[fan]
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3 [fan] 按照VGG的设计，每个Stage会按两倍率下采样，同时可以发现不同Stage的卷积核的个数乘了2
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################
def apply_box_deltas_graph(boxes, deltas): # 利用预测的回归值，对框进行微调[fan]
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    #[1] 将box的边界表示法转换为中心表示法
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    #[1] 根据deltas精修锚框 # 这里根据文档的资料反推，可以对上[fan]
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    #[1] 将精修后的中心表示法的box转换为边界表示法
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip # 砍掉超出限定边界的边框区域。以图片左上角为(0,0)。minimum()表明限定框的坐标值最大不可大过限定右下角的坐标值。maximum()表明限定框的坐标值最小不可小过限定左上角的坐标值 [fan]
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    fan: 下面的batch应该是同时计算的图片的个数。
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, fg prob]
        scores = inputs[0][:, :, 1] # 所有Anchor前景的概率 要看上面注释提供的维度信息好理解些 [fan]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1] # Anchor坐标微调 [fan]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # [fan] simple Faster R-CNN中也出现了方差。 乘方差的原因是，为了将delta 恢复成 坐标，以便于后面做NMS计算交并比。  请见 https://github.com/matterport/Mask_RCNN/issues/1900

        # Anchors
        anchors = inputs[2] # Anchor的坐标 [fan]

        # Improve performance by trimming(修剪) to top anchors by score
        # and doing the rest on the smaller subset. 框太多了，在做NMS前先根据框的分数选择一小部分框。PRE_NMS_LIMIT = 6000[fan]
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1]) # min(6000,num_anchors) [fan]
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices  # 按照scors排名，保留前pre_nms_limit个anchors的index [1]
        # (多GPU)取出前景分数靠前的框的 前景概率、RPN位置回归值、Anchor坐标值 [fan]
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]# 利用RPN预测的回归值，对得分靠前的Anchor进行微调[fan]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32) # 坐标值的限定框.因为回归没有做sigmoid这类激活函数,预测值可能会超出(0,1)之间 [fan]
        boxes = utils.batch_slice(boxes, # 砍掉超出限定边界的边框区域 [fan]
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes(这里在simple Faster R-CNN的代码中有，删掉了少于16*16的框 [fan])
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.
        # for small objects, so we're skipping it.

        # Non-max suppression [4] 执行非极大值抑制，根据IoU阈值选择出2000个rois，如果选择的rois不足2000，则用0进行pad填充
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,# 参数三为最大返回数目[1]
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            # 一旦返回数目不足, 填充(0,0,0,0)直到数目达标 [1]
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals #[4] 最终返回的proposals赋值给rpn_rois，作为rpn网络提供的建议区，注入后续FPN heads进行分类、目标框和像素分割的检测。

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0) # [fan] log换底公式 https://zhidao.baidu.com/question/497044639399676604.html

# [1] PyramidROIAlign用于提取rois区域特征，输出维度为[batch, num_boxes, 7,7,256]
class PyramidROIAlign(KE.Layer):
    # [3] 我们需要按照RCNN的思路，使用proposal对共享特征进行ROI操作，在Mask-RCNN中这里有两个创新： 1.ROI使用ROI Align取代了之前的ROI Pooling；2.共享特征由之前的单层变换为了FPN得到的金字塔多层特征，即：mrcnn_feature_maps = [P2, P3, P4, P5]。创新点2意味着我们不同的proposal对应去ROI的特征层并不相同，所以，我们需要：按照proposal的长宽，将不同的proposal对应给不同的特征层；在对应特征层上进行ROI操作。这个class基本实现了我们开篇所说的全部功能，即特征层分类并ROI。
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        # [1]
        # num_boxes指的是proposal数目，它们均会作用于每张图片上，只是不同的proposal作用于图片
        # 的特征级别不同，我通过循环特征层寻找符合的proposal，应用ROIAlign
        boxes = inputs[0] # [fan] 提议框的坐标(已归一化)

        # Image meta # [fan] 图像信息
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:] # [fan] 不同分辨率的特征图

        # Assign each ROI to a level in the pyramid based on the ROI area. [fan] 根据论文的公式 和 提议框的面积 将提议框分配到特定的特征图
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2) # [fan] 提议框的坐标(已归一化) [batch, num_boxes, (y1, x1, y2, x2)] ->  [batch, num_boxes, 1]
        h = y2 - y1 # [fan] 框高
        w = x2 - x1 # [fan] 框宽
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0] # [1]图像的 h, w, c
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4 # [fan] 如果RoI面积是224*224，那么它应该分配到P4特征图
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))) #[fan] 其实相当于 log_{2}(tf.sqrt(h*w/image_area)/224.0)。如果tf.sqrt(h*w/image_area)=224，那么这一项为0，后面的roi_level=4，即对应到P4特征图
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32))) # [1] 确保值位于2到5之间
        roi_level = tf.squeeze(roi_level, 2)# [fan] 这里已经将提议框匹配到对应的特征图了(但实际上还没有求RoI的特征) [1,?,1]->[1,?] 该函数返回一个张量，这个张量是将原始input中第二维度为1的删掉 https://www.jianshu.com/p/a21c0bc10a38

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)): # [fan] level是特征图的序号，从2到5
            ix = tf.where(tf.equal(roi_level, level)) # [fan] 之前给提议框绑定特征图了，现在根据提议框的特征图序号选出 与当前循环的特征序号相同的 提议框(的序号) 。 ix的形状是(?,2),维度1是True的个数，维度2是坐标。tf.euqal广播返回[1,?]， tf.where返回[num_true(?), dim_size(condition)(2)]
            level_boxes = tf.gather_nd(boxes, ix) # [fan] 当前level的提议框的坐标(已归一化) # [1] [本level的proposal数目, 4] # 高级高维切片gather_nd https://www.cnblogs.com/hellcat/p/9819697.html

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32) # [1] 记录每个propose对应图片序号 [fan] 这里的维度变化我暂时放弃

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes) # [fan]把它们当成常量，不希望RPN改动？ https://github.com/matterport/Mask_RCNN/issues/343
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin, # [fan] 这里是一个bin(桶)里面只采样一个点,见论文第4页左上角第一段 [fan]
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape, # [fan] level_boxes是提议框(归一化)的坐标，即从特征图中选择提议框对应的区域做双线性插值，使得不同大小的提议框区域的特征图统一缩放到pool_shape*pool_shape大小
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0) # [3] [batch*num_boxes, pool_height, pool_width, channels]

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0) # [3] [batch*num_boxes, 2]
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1) # [3] [batch*num_boxes, 1]
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1) # [3] [batch*num_boxes, 3]

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        # [1]
        # 截止到目前，我们获取了记录全部ROIAlign结果feat集合的张量pooled，和记录这些feat相关信息的张量box_to_level，
        # 由于提取方法的原因，此时的feat并不是按照原始顺序排序（先按batch然后按box index排序），下面我们设法将之恢复顺
        # 序（ROIAlign作用于对应图片的对应proposal生成feat）
        # [3] box_to_level[i, 0]表示的是当前feat隶属的图片索引，box_to_level[i, 1]表示的是其box序号
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1] # [3] [batch*num_boxes]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        # [3] [batch, num_boxes, (y1, x1, y2, x2)], [batch*num_boxes, pool_height, pool_width, channels]
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled # [3] [batch, num_boxes, pool_height, pool_width, channels]

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops. # [fan] boxes1与boxes2 统一维度，这样就不用使用for循环
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections [fan]计算交集的四个坐标和面积
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions [fan]计算并集的面积
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2] [fan]计算交并比
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]]) # [fan] boxes1与boxes2不是两个框，而是两种许多框；某个boxes1要与所有的boxes2计算交并比，这里将交并比整理成[boxes1, boxes2]的形式，方便理解
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    # [5] 获取的就是每个rois和哪个真实的框最接近，计算出和真实框的距离，以及要预测的mask，这些信息都会在网络的头的classify和mask网络所使用
    # [fan] 如果要对Head进行训练，不仅需要对RPN生成的提议框下采样以减少数量(2000->200)，还需要我们自己生成每个RoIs的类别真值(应当属于哪个类别target_class_ids)、回归真值(与目标框的回归偏移量是多少target_bbox)、RoIs的掩模 , 原始数据提供的仅有RoIs的边界框(target_rois)、原图中物体的边界框(gt_boxes)、原图中物体的类别(input_gt_class_ids)、物体的掩模Mask(其个数是物体的个数 input_gt_masks)
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.# [fan] target_rois/proposals RPN提议的边界框
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs # [fan] input_gt_class_ids 原图中物体的类别
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates. # [fan] gt_boxes 原图中物体的边界框
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type. # [fan] input_gt_masks 物体的Mask(其个数是物体的个数，深度只有1，即只取真实类别的掩模)

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks. # [fan] RoIs的掩模真值(target_mask)
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates # [fan] 这里应该是数量有减少筛选后的提议框
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.# [fan] 每个RoIs的类别真值(应当属于哪个类别)
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))] # [fan] 回归真值(与目标框的回归偏移量是多少)
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox # # [fan] RoIs的掩模真值(target_mask)
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], # 比较大小 https://www.jianshu.com/p/e227effbc9ac
                  name="roi_assertion"), # tf.Assert() 根据条件打印数据   https://blog.csdn.net/u013066730/article/details/98478473
    ]
    # 解释：对于control_dependencies这个管理器，只有当里面的操作是一个op时，才会生效，也就是先执行传入的参数op，再执行里面的op。而y=x仅仅是tensor的一个简单赋值，不是定义的op，所以在图中不会形成一个节点，这样该管理器就失效了。tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中，这时control_dependencies就会生效 https://blog.csdn.net/hu_guan_jie/article/details/78495297
    with tf.control_dependencies(asserts): # [fan] 先执行asserts这个op：如果proposals的个数不大于0，打印proposals信息；
        proposals = tf.identity(proposals) # [fan] 然后执行op:复制 (需要是op才有效)

    # Remove zero padding # [fan] 去除坐标全零的框，也就是只留下真实存在的有意义的框
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, # [fan] 上面判断框是否全零应删除，生成了一个bool值的序列，这里顺带利用这个序列，删除类别和掩模
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")# [1] 根据tf.where(non_zeros)[:, 0]从gt_masks中获取非零物体边框对应的masks

    # Handle COCO crowds 处理COCO训练集中标记为拥挤的框[fan]
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID. # [fan] 拥挤的框是那种 一个框的附近有很多物体的框。这种框在训练时需要排除。这些框的类别是一个负数。
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0] # [fan] 拥挤的框序号列表
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0] # [fan] 非拥挤的框序号列表
    crowd_boxes = tf.gather(gt_boxes, crowd_ix) #[fan] 拥挤的框的坐标
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix) #[fan] 非拥挤的框的类别
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix) #[fan] 非拥挤的框的坐标
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2) #[fan] 非拥挤的框对应物体的掩模

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes) # [fan] 计算每个RPN提议框和每个gt_boxes的IoU

    # Compute overlaps with crowd boxes [proposals, crowd_boxes] 
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes) # [fan] 计算每个RPN提议框和每个拥挤框的IoU
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1) #[fan] 每个RPN提议框对应了多个物体框的交并比，取与提议框交并比最大的框作为它分配的真值物体框
    no_crowd_bool = (crowd_iou_max < 0.001) # [fan] 这里自定义地进一步地细分拥挤框，认为提议框与拥挤框交并比小于0.001的提议框仍然可以作为负类用于训练。

    # Determine positive and negative ROIs 确定RoIs的正类负类
    roi_iou_max = tf.reduce_max(overlaps, axis=1) #[fan] 每个RPN提议框对应了多个物体框的交并比，取与提议框交并比最大的框作为它分配的真值物体框
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box # [fan] RPN提议框与物体框交并比大于等于0.5的提议框视为正类(没有考虑是否拥挤，即不管框内是否有有其他物体)
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds. # [fan] RPN提议框与物体框交并比小于0.5 且 RPN提议框与拥挤框交并比小于0.001 的提议框视为负类(考虑了是否拥挤，即最好框内不要有物体，这符合负类即背景类的定义)
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs #[fan] RPN提议框下采样2000->200(如果数目不足最后会补全0样本，如果正类小于66个，那就不是200个，但是要保证比例是1:3). Aim for 33% positive #[fan] 正类/负类 =1/3
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO) # 200*0.33 取整66，正样本个数 [fan]
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count] # [fan] 从positive_indices中随机挑选66个。而不是取分数最高或最低的，存疑吧。
    positive_count = tf.shape(positive_indices)[0] # [fan]实际上正类的数目可能小于66，这里取shape可以得到真实的数目
    # Negative ROIs. Add enough to maintain positive:negative ratio. # [fan] 保证正负比例为1:3。 不考虑负样本不足的情况么 存疑？
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs  #[fan] 收拢选中的提议框坐标
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)
    #-----------上面是根据IoU选择提议框----------------fan-------------下面是对选中的提议框绑定真值--------------#
    # Assign positive ROIs to GT boxes. #[fan] 给选中的正类提议框 绑定上物体边框(类别和回归偏移量)
    positive_overlaps = tf.gather(overlaps, positive_indices) #[fan] 收拢选中的正类提议框与每个物体框的交并比(还没有求最大)
    roi_gt_box_assignment = tf.cond( #[fan] 条件判断,得到正类提议框对应的物体框索引。 可参考 https://blog.csdn.net/TeFuirnever/article/details/88875727
        tf.greater(tf.shape(positive_overlaps)[1], 0),#[fan] 条件：判断positive_overlaps([boxes1, boxes2]) 的形状是否异常 。
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1), #[fan] 如果形状正常，则依据最大交并比，得到正类提议框对应的物体框索引
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64) #[fan] 如果形状不正常，返回空
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment) #[fan] 得到正类提议框对应的物体框的四个坐标值
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment) #[fan] 得到正类提议框对应的物体框的类别

    # Compute bbox refinement for positive ROIs # [5]用最接近的真实框修正rpn网络预测的框 #[1]依据roi_gt_boxes对positive_rois进行修正，是与gt_box的差异 #[fan] 计算由提议框变化到物体框的偏移量，作为边框回归的真值
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV #[fan] 除以方差，simple Faster R-CNN中也有这样的操作。之前RPN的回归预测值是乘以框的方差，我觉得那里是乘有问题。存疑。提了问题：  https://github.com/matterport/Mask_RCNN/issues/1900

    # Assign positive ROIs to GT masks #[fan] 给选中的正类提议框 绑定对应物体的掩模
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1) #[fan] transpose  [height, width, MAX_GT_INSTANCES]-->[MAX_GT_INSTANCES,height, width]  https://blog.csdn.net/weixin_36396470/article/details/86515169
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment) #[fan] 取提议框所绑定的物体的掩模。一张图片有多少个物体，那就会有多少个Mask。

    # Compute mask targets
    boxes = positive_rois #[fan] 正类提议框的坐标
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        # [fan] 如果采用mini_mask模式，输入的roi_masks就是从全图的Mask中截取物体框区域的掩模并做缩放，需要先求提议框与目标框的交集，然后求交集边界两点在目标框中对应的坐标，根据这两个坐标在物体掩模中截取RoI的掩模
        # [4]
        # 如果采用mini_mask模式，则需要在这里将positive_rois转换到roi_gt_boxes的范围内,
        # 因为在mini_mask模式下，仅仅记录了gt_boxes(某个物体框)内部的mask信息
        # 正如作者解释注释的＂We store mask pixels that are inside the object bounding box,
        #rather than a mask of the full image.Most objects are small compared to the image size, so we save space by not storing a lot of zeros around the object.＂
        # 作者注释可参考./samples/coco/inspect_data.ipynb的Mini_Masks部分
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1) # 注意这里是positive_rois的坐标值 [fan]
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1) # 这里是RoI对应的物体框的坐标值 [fan]
        gt_h = gt_y2 - gt_y1 # 物体框的高 [fan]
        gt_w = gt_x2 - gt_x1 # 物体框的宽 [fan]
        y1 = (y1 - gt_y1) / gt_h # [fan]以物体框的左上角点(gt_x1,gt_y1)为(0,0)，求RoI与物体框的交集的左上角(x1,y1)和右下角坐标(x2,y2)，然后还需要用物体框的长和高 将坐标值归一化(后面的截取函数需要)。
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    # [fan] 如果不使用MINI_MASK，则输入的roi_masks是全图大小的Mask，直接从全图Mask中截取RoI对应区域的掩模，作为RoI的masks 。 无论是物体掩模的RoI交集区域Mask，还是全图掩模的RoI区域的Mask，我觉得都是比较合理的，因为得到的掩模并没有超出RoI能覆盖的区域。

    # [4] 设定每个box对应的id
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    # [4] 采用tf.image.crop_and_resize根据boxes，box_ids在roi_masks上截取并重采样至28*28，这样mask分支才能正常训练. # [fan] 缩放的方法默认是双线性插值,所以需要是浮点数
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, # 参考  https://blog.csdn.net/m0_38024332/article/details/81779544
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3) #[fan]  [N, height, width, 1] -》 [N, height, width]

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks) # [fan] 因为是双线性插值，所以会有浮点数。如果>=0.5为1，如果<0.5为0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    #[1] 最后如果rois不足200,则用0进行pad填充 #[fan] 从上面的代码看，先确定正类的个数，再根据1:3比例，确定负类的个数，此时负类不一定有那么多 或者 总数不一定能凑足200个，此时就需要用全0来凑数，我觉得全0没有什么实际意义，只是用来凑数。
    rois = tf.concat([positive_rois, negative_rois], axis=0) #[fan] 合并选中的正负提议框坐标
    N = tf.shape(negative_rois)[0] #[fan] 负类的个数
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0) #[fan] RoI需要补P才到200
    rois = tf.pad(rois, [(0, P), (0, 0)]) #[fan] pad是对每一维度的最前和最后进行填充，这里是在后面填充P行的0. 参考：https://blog.csdn.net/qq_40994943/article/details/85331327
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)]) #[fan] 坐标框补N+P(负类+凑数的)行的0
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)]) #[fan] 类别补N+P(负类+凑数的)行的0
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)]) #[fan] 回归偏移量补N+P(负类+凑数的)行的0
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)]) #[fan] 掩模补N+P(负类+凑数的)行的0

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each. # [fan] 如果要对Head进行训练，不仅需要对RPN生成的提议框下采样以减少数量(2000->200)，还需要我们自己生成每个RoIs的类别真值(应当属于哪个类别target_class_ids)、回归真值(与目标框的回归偏移量是多少target_bbox)、RoIs的掩模 , 原始数据提供的仅有RoIs的边界框(target_rois)、原图中物体的边界框(gt_boxes)、原图中物体的类别(input_gt_class_ids)、物体的掩模Mask(其个数是物体的个数 input_gt_masks)

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals. # 提议框数目不够，用零来凑[fan]
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs. # 目标框-类别真值 ,MAX_GT_INSTANCES应该是图中物体的个数[fan]
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized # 目标框-坐标真值 [fan]
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type # 物体的Mask（其个数是物体的个数) [fan]

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks. # 返回每个ROIs应当属于哪个类别、与目标框的回归偏移量是多少、RoIs的掩模 [fan]
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates # 每个RoIs的坐标值 [fan]
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs. # 每个ROIs应当属于哪个类别 [fan]
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)] # 每个ROIs与目标框的回归偏移量是多少[fan]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width] # 每个RoIs的掩模[fan]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs): # inputs:target_rois, input_gt_class_ids, gt_boxes, input_gt_masks [fan]
        proposals = inputs[0] # target_rois/proposals RPN提议的边界框[fan]
        gt_class_ids = inputs[1] # input_gt_class_ids 原图中物体的类别[fan]
        gt_boxes = inputs[2] # gt_boxes 原图中物体的边界框[fan]
        gt_masks = inputs[3] # input_gt_masks 物体的Mask(其个数是物体的个数)[fan]

        # Slice the batch and run a graph for each slice 多GPU[fan]
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks], # 输入 [fan]
            lambda w, x, y, z: detection_targets_graph( # [fan] RoIs筛选到200个 和生成RoIs的类别分类、边框回归、分割真值 的具体实现
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.# [fan:] 根据Head预测的分类和回归输出，对2000个RoIs进行筛选，剔除掉RoIs中预测概率低的、预测为背景的、冗余高度重叠的，然后根据最高分类概率取top-k个RoIs的偏移量，RoIs微调后会作为Mask分支的输入

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    [3]
    注意，下面调用的函数，每次处理的是单张图片。
    逻辑流程如下：
    a 获取每个推荐区域得分最高的class的得分
    b 获取每个推荐区域经过粗修后的坐标和"window"交集的坐标
    c 剔除掉最高得分为背景的推荐区域
    d 剔除掉最高得分达不到阈值的推荐区域
    e 对属于同一类别的候选框进行非极大值抑制
    f 对非极大值抑制后的框索引：剔除-1占位符，获取top k（100）
    最后返回每个框(y1, x1, y2, x2, class_id, score)信息
    """

    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32) # [3:]  [N]，每张图片预测的最高得分类
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1) # [3:]  [N, (图片序号, 预测的最高class序号)]
    class_scores = tf.gather_nd(probs, indices) # [3:]  [N], 每张图片最高得分类得分值
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices) # [fan]  每张图片预测的最高得分类对应的偏移量
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(  # 利用预测的回归值，对框进行微调，微调后的框作为Mask分支的输入[fan]
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window) # [3] b 获取每个推荐区域经过粗修后的坐标和"window"交集的坐标

    # TODO: Filter out boxes with zero area

    # Filter out background boxes # [3] c 剔除掉最高得分为背景的推荐区域
    keep = tf.where(class_ids > 0)[:, 0] #  [3] class_ids: N, where(class_ids > 0): [M, 1] 即where会升维
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE: # [3] d 剔除掉最高得分达不到阈值的推荐区域 # 0.7
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        # [3]
        # 求交集，返回稀疏Tensor，要求a、b除最后一维外维度相同，最后一维的各个子列分别求交集.[fan] RoIs交集 ： 最高得分不能为背景类 同时 最高得分要超过阈值
        # a   = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
        # b   = np.array([[{1}   , {}] , [{4}, {5, 6, 7, 8}]])
        # res = np.array([[{1}   , {}] , [{4}, {5, 6}]])
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0] # [3]  此时使用张量keep保存符合条件的推荐区域的index，即一个一维数组，每个值为一个框的序号，后面会继续对这个keep中的序号做进一步的筛选。

    # Apply per-class NMS  # [3] e 对属于同一类别的候选框进行非极大值抑制 # [fan] RPN后的那次NMS没有分类别地处理，因为只有一类：前景类有坐标
    # 1. Prepare variables [3] 这一部分代码主要对于当前的信息进行整理为精炼做准备
    pre_nms_class_ids = tf.gather(class_ids, keep) # [3:]  [n]
    pre_nms_scores = tf.gather(class_scores, keep) # [3:]  [n]
    pre_nms_rois = tf.gather(refined_rois,   keep) # [3:]  [n, 4]
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0] # [3:]  去重后class类别
    '''
    # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    y, idx = unique(x)
    y ==> [1, 2, 4, 7, 8] [fan] 这里是输出类似这个的东西 
    idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    '''
    # [3] 注意下面的内嵌函数，包含keep（step1中保留的框索引）、pre_nms_class_ids（step1中保留的框类别）、pre_nms_scores（step1中保留的框得分）几个外部变量，
    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        # [3] class_id表示当前NMS的目标类的数字，pre_nms_class_ids为全部的疑似目标类
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs), # [3] 当前class的全部推荐区坐标
                tf.gather(pre_nms_scores, ixs), # [3]  当前class的全部推荐区得分
                max_output_size=config.DETECTION_MAX_INSTANCES, # [3]  100
                iou_threshold=config.DETECTION_NMS_THRESHOLD)  # [3]  0.3
        # Map indices
        # [3] class_keep是对ixs的索引，ixs是对keep的索引
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep)) # [3] 由索引获取索引
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0] # [fan] 每个类别的RoIs默认100个不足补-1
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        # [3] 返回长度必须固定，否则tf.map_fn不能正常运行
        return class_keep

    # 2. Map over class IDs  # [3]  e 对属于同一类别的候选框进行非极大值抑制。
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64) # [3：]  [?, 默认100]：类别顺序，每个类别中的框索引，？表示该张图片中保留的类别数（不是实例数注意)
    # 3. Merge results into one list, and remove -1 padding
    #[3] f 对非极大值抑制后的框索引：剔除 - 1 占位符
    nms_keep = tf.reshape(nms_keep, [-1]) #[3] 全部框索引
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0]) #[3] 剔除-1索引
    # 4. Compute intersection between keep and nms_keep
    # [3] nms_keep本身就是从keep中截取的，本步感觉冗余
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep) # [3] 获取得分
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1] # [fan] 根据预测的概率 获取top k（100）
    keep = tf.gather(keep, top_ids) # [3] 由索引获取索引

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),  # [3]  索引坐标[?, 4]
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],  # [3]  索引class，添加维[?, 1]
        tf.gather(class_scores, keep)[..., tf.newaxis]  # [3]  索引的分，添加维[?, 1]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    # [3] 如果 detections < DETECTION_MAX_INSTANCES，填充0
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size. fan:这份代码设置所有图像大小一致
        m = parse_image_meta_graph(image_meta) # [fan]: 得到图片信息 [3] 用于解析并获取输入图片的shape和原始图片的shape（即"window"）
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2]) # [3] (y1, x1, y2, x2). 我们经过"window"获取了原始图片相对输入图片的坐标（像素空间），然后除以输入图片的宽高，得到了原始图片相对于输入图片的normalized坐标，分布于[0,1]区间上。 事实上由于anchors生成的4个坐标值均位于[0,1]，在网络中所有的坐标都是位于[0,1]的，原始图片信息是新的被引入的量，不可或缺的需要被处理到正则空间。


        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU) # [fan:] 根据Head预测的分类和回归输出，对1000个RoIs进行筛选，剔除掉RoIs中预测概率低的、预测为背景的、冗余高度重叠的，然后根据最高分类概率取top-k个RoIs

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN # 滑动窗口 [fan]
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)
    # 分类[fan]
    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)
    # 回归[fan]
    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox] # rpn_class_logits是由rpn_class_logits经过softmax后产生的 [fan]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################
# [4] 首先利用PyramidROIAlign提取rois区域的特征，再利用TimeDistributed封装器针对num_rois依次进行7*7->1*1卷积操作，再分出两个次分支，分别用于预测分类和回归框。
def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities # [fan] 经过了softmax的
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    # [1] PyramidROIAlign将pool_size划分的7*7区域(对于mask_pool_size则为14*14)取若干采样点后，
    # [1] 进行双线性插值得到f(x,y)
    # [4] PyramidROIAlign层不展开讨论，可认为将pool_size划分的7 * 7
    # [4] 区域(对于mask_pool_size则为14 * 14) 取若干采样点后，进行双线性插值得到f(x, y)，这个版本的代码中取采样点为1。
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # [3] 经过ROI之后，我们获取了众多shape一致的小feat，为了获取他们的分类回归信息，我们构建一系列并行的网络进行处理
    # Two 1024 FC layers (implemented with Conv2D for consistency) # [fan] 这里的卷积核大小与特征图大小一样大(pool_size*pool_size)，卷积核个数与全连接神经元个数一样(1024)，这里的卷积层 等价于 全连接层 http://cs231n.github.io/convolutional-networks/#convert
    # [1] conv(1024,7,7)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), # TimeDistributed 把输入的第一个维度看成是batch(时间)，在相同的层上运算，每个batch(时间)的计算是独立的 https://github.com/matterport/Mask_RCNN/issues/1118#issuecomment-438489399
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [1] conv(1024,1,1)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), # [fan] 这里的卷积核大小与特征图大小一样大(1*1)，卷积核个数与全连接神经元个数一样(1024)，这里的卷积层 等价于 全连接层 http://cs231n.github.io/convolutional-networks/#convert
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [1] [batch, num_rois, 1024]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head # 全连接层 神经元个数是81 即类别数(包含背景) ，使用softmax
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head # [fan] 全连接层 神经元个数是81*4 即类别数*4(包含背景) ，使用softmax
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox # [fan] 框分类，框回归

# [1]
# 利用PyramidROIAlign提取rois区域的特征，
# 再利用TimeDistributed封装器针对num_rois依次进行3*3--->3*3--->3*3--->3*3卷积操作，
# 再经过2*2的转置卷积操作，得到像素分割结果。
def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    # [1] PyramidROIAlign用于提取rois区域特征，输出维度为[batch, num_boxes, 14, 14, 256]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers # [4] 4层常规3*3卷积层整合特征
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    # [4]  1层转置卷积进行上采样，将特征层扩大2倍
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    # [4] 最终输出Masks做为分割结果，维度为[batch, num_rois, 28, 28, 81]，这里为每一类实例都提供一个channel，原论文的观点是"避免了不同实例间的种间竞争"。
    # [fan] 这里的预测输出 与 之前生成的真值roi mask 长宽是一致的；前者的深度为num_classes，后者的深度为1
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), # [fan] sigmoid，保证每个像素位置介于01之间
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred) # fan: 绝对值
    less_than_one = K.cast(K.less(diff, 1.0), "float32") # fan: 阈值
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5) # fan: 两个函数
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    # [fan] rpn_match 是所有anchor的类别真值；rpn_class_logits 是所有anchor的类别预测值
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor. # [fan] anchor类别真值。0应该代表损失无关的填充
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG. # anchor类别预测值
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32) # [fan] 将样本的标签值由-1、0、1转为0/1,之前填充的0也包括在变化之后的0中，但是后面只会提取正负样本，也就排除了中立的样本
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0)) # [1] 提取不等于0的样本对应index,即是提取正负样本
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices) # [1] anchor_class 对应标签值为0/1的负/正样本
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class, # [1] 调用的是sparse_softmax_cross_entropy_with_logits
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0)) # [fan] if loss个数大于0, 平均；else 为0.0
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    # [fan] 根据target_bbox与rpn_bbox计算anchors的回归损失
    # [fan] 输入：  target_bbox是选中的anchor的回归真值；rpn_match是所有anchor的类别真值;rpn_bbox是所有anchor的预测回归值
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors(fan:这里指每个批次中正类Anchors个数), (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't. # [1] 只计算正样本的bbox损失
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1)) # [fan] indices只记录了选定的正类(有物体/前景)Anchors的回归预测值 的索引

    # Pick bbox deltas that contribute to the loss #
    rpn_bbox = tf.gather_nd(rpn_bbox, indices) # [fan] 取 正类(有物体、前景)Anchors的回归预测值  [batch, anchors, (dy, dx, log(dh), log(dw))] =》[anchors, (dy, dx, log(dh), log(dw))]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1) # [fan] 每个batch中正类anchor的个数
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU) # [fan] [batch, max positive anchors, (dy, dx, log(dh), log(dw))] =》 [max positive anchors, (dy, dx, log(dh), log(dw))]

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0)) # [fan] if loss个数大于0, 平均；else 为0.0
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array. # [fan]  RoIs的类别真值(应当属于哪个类别target_class_ids) ;
    pred_class_logits: [batch, num_rois, num_classes]  # [fan]   mrcnn_class_logits 是RoIs的类别预测值(softmax前)
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.  # [fan]  涉及到active_class_ids，即将该图片隶属数据集中所有的class标记为1，不隶属本数据集合的class标记为0，计算Loss贡献时交叉熵会对每个框进行输出一个值，如果这个框最大的得分class并不属于其数据集，则不计本框Loss [3]
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2) # [fan] 取预测概率最大的那个类
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids) # [fan] 每个roi预测的类别有没有(是不是active）就知道了 https://zhuanlan.zhihu.com/p/45673869

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active # [3] 涉及到active_class_ids，即将该图片隶属数据集中所有的class标记为1，不隶属本数据集合的class标记为0，计算Loss贡献时交叉熵会对每个框进行输出一个值，如果这个框最大的得分class并不属于其数据集，则不计本框Loss

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    # [fan] RoIs的回归真值(RoIs与目标框的回归偏移量是多少target_bbox) ； RoIs的类别真值(应当属于哪个类别target_class_ids，包括负类0) ; pred_bbox 是Head预测的RoIs的回归偏移量
    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.[fan:] 简单来说就是batch*num_rois
    target_class_ids = K.reshape(target_class_ids, (-1,)) # [3] [batch, num_rois] -》 [batch*num_rois]
    target_bbox = K.reshape(target_bbox, (-1, 4)) # [fan] [batch, num_rois, (dy, dx, log(dh), log(dw))] -》 [batch * num_rois, (dy, dx, log(dh), log(dw))]
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4)) # [fan] [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))] -》 [batch*num_rois, num_classes, (dy, dx, log(dh), log(dw))]

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices. # 只用正类(有物体) 的RoIs 计算回归损失 [fan]
    # class_ids: N, where(class_ids > 0): [M, 1] 即where会升维 [3]
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0] # [M] # [fan:] 正类(有物体)的索引
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64) # [fan] 根据正类的索引取到RoIs真值 正类的类别

    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1) # [(正类框序号，正类真实类别id)，……] [3]

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix) # [fan] 根据正类的索引取到RoIs真值 正类偏移量
    pred_bbox = tf.gather_nd(pred_bbox, indices) # 从预测的多个类的回归值中，取出真值类别对应的预测回归值 [fan]

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), # 正类的RoIs的类别真值对应的回归预测值  与  回归真值 计算损失[fan]
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    # [fan] target_mask  每个RoIs的掩模 ; 每个RoIs的类别真值(应当属于哪个类别target_class_ids) ; pred_masks:预测的每个RoIs的每个类别的Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix) # [fan] 取真值中的 正类RoIs的mask
    y_pred = tf.gather_nd(pred_masks, indices) # [fan] 取预测中的 正类RoIs的真实类别的mask

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred), # [3] keras的二进制交叉熵实际调用的就是sigmoid交叉熵的后端 https://www.cnblogs.com/hellcat/p/8568005.html
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    # [fan] 训练RPN网络时，设定RPN训练样本总数是256个，根据anchor与物体框的交并比，确定Anchor的类别(正类大于0.7、负类小于0.3、与损失不相关的中立类)，从正类中随机选择，并使得样本中正负样本的比例为1：1，然后求正类anchor到物体框的偏移量
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32) # [fan] 注意这里的类别 是所有锚点anchor的类别，没有做筛选
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4)) # [fan] 注意这里的偏移量 包括了正类和负类以及无关类，但总数不是所有的anchor数，而是config.RPN_TRAIN_ANCHORS_PER_IMAGE，正类和负类的偏移量直接为0

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from 一个对象
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True: # 生成批次的数据，一张图片为一个批次 [fan]
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets # [fan] 训练RPN网络时，根据anchor与物体框的交并比，确定Anchor的类别(正类、负类、与损失不相关的类)，确定RPN训练的样本的个数(随机选择)，并使得样本中正负样本的比例为1：1，然后求正类anchor到物体框的偏移量
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays 一个batch的数据(一张图片一张图片地充能，直到达到一个batch_size就输出)
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
            # 如果一张图皮中的物体超出了限制，随机取
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir() # 设置 从以前训练的地方开始训练 或者 从头开始训练，设置模型保存的地址[fan]
        self.keras_model = self.build(mode=mode, config=config) # 构建Mask R-CNN模型[fan]

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.训练 推理 的输入输出尺寸不一致 [fan]
        """
        assert mode in ['training', 'inference'] # 不是这两种字符串会报错[fan]

        # Image size must be dividable by 2 multiple times
        # 输入图像必须能被2的6次方64整除 [1]
        # 這個網路只接受長寬皆是64倍數的圖片，原因在於經過ResNet以及稍後MaskRCNN裡出現的MaxPooling2D之後，feature map的長寬會變為原來的1/64，為了避免出現不整除的情況，才會有此規定 [2]
        h, w = config.IMAGE_SHAPE[:2] # 规定、设置 : 长宽需要能被2的6次方64整除[fan]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        # 训练和测试阶段都有的输入 [fan]
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        # 图片的信息（包含形状、预处理信息等） [1]
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # 训练阶段特有的输入，训练RPN和Faster_RCNN需要的真值 [fan]
            # RPN GT
            # RPN的输入之一 : Anchor的类别 [batch(批次个数) ,None(某个批次中所有Anchor的个数), 1(1代表是物体，-1代表是背景，0代表与损失无关，)] [fan]
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            # RPN的输入之一 : 正类Anchor的回归真值(偏移量)  [batch(批次个数) ,None(某个批次中选中的用于训练的Anchor个数), 4(四个坐标值，这里的偏移量 包括了正类和负类以及无关类，正类和负类的偏移量直接为0)] [fan]
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates 坐标归一化 [fan]
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK: # True:只会截取物体框区域的mask，然后缩放到更小的尺寸(56,56)，以减小内存负载。 [fan]
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else: # False，否则Mask的长宽就是原图像的大小 [fan]
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # [3]： inference模式中，整个程序需要外部输入的变量仅有三个，注意keras的习惯不同于placeholder，下面代码的shape没有包含batch，实际shape是下面的样式：
            # input_image：输入图片，[batch, None, None, config.IMAGE_SHAPE[2]]
            # input_image_meta：图片的信息（包含形状、预处理信息等，后面会介绍），[batch, config.IMAGE_META_SIZE]
            # input_anchors：锚框，[batch, None, 4]

            # 测试阶段特有的输入[fan]：
            # Anchors in normalized coordinates # [1:] input_anchors：锚框，[batch, None, 4]
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # [fan] 特征提取、RPN输出下采样(NMS等) 训练和测试过程是一样一样的
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.

        # [1]: config类定义于mrcnn/config.py文件中，这里配置了模型中的众多超参数，会经常看到
        # config.BACKBONE是用于选择使用何种网络结构，这里使用的是resnet101
        # config.TOP_DOWN_PYRAMID_SIZE是用于规定FPN网络时的通道数，这里是256
        # resnet_graph是用于定义resnet函数
        # 如果不是resnet50、resnet101，就要重新计算
        if callable(config.BACKBONE): # 需要定义了__call__的对象才为True [fan]
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else: # 经由如下判断（inference中该参数是字符串"resnet101"，所以进入else分支），建立ResNet网络图[3]
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, # 这个函数在文档的"构建ResNet"中详细说明了 [fan]
                                             stage5=True, train_bn=config.TRAIN_BN) # if config.TRAIN_BN is False , Freeze BN layers. Good when using a small batch size
        # Top-down Layers # 文档里FPN部分详细描述了。因为要共享RPN等，所以特征图的通道数需要一致，这里是256[fan]
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5) #
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5) # P……

        # Note that P6 is used in RPN, but not in the classifier heads. # [fan]这里的P6是对P5简单地下采样得到的，仅仅是为了在RPN中和512^2面积大小的anchor对应，P6不会在之后的Fast R-CNN中使用
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors # 生成FPN各个特征图的锚框，不同分辨率的特征图各自对应了一种大小的Anchor(但形状有三种) [fan]
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around(避开) Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model 文档里详细描述了 (这里并不包括计算RPN损失的过程,也即这里没有考虑训练RPN的过程，也即这里没有考虑筛选anchor样本生成真值的过程) [fan]
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers # 每个特征图(特征金字塔)共享同一个RPN [fan]
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]] !!!!!!!!
        # 将每一层的输出抽取出来聚合在一起 [1]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"] # [fan] 所有anchor的softmax后的预测类别概率，所有anchor的softmax前的预测类别概率，所有anchor回归预测值
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]
        # [1]:
        # [batch, num_anchors, 2/4]
        # 其中num_anchors指的是全部特征层上的anchors总数
        # 2是class的维度， 4是bbox的维度 # 这里描述的应该是下面三个RPN输出的维度 [fan]
        rpn_class_logits, rpn_class, rpn_bbox = outputs # rpn_class是由rpn_class_logits经过softmax处理过的 [fan]

        # Generate proposals # RPN区域提议和筛选，生成RoIs [fan]
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        # Proposal得到的roi不够config.POST_NMS_ROIS_TRAINING时，在NMS中用0补齐 [1]
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE # 训练时NMS后剩余2000个框(测试时NMS后剩余1000个框)[fan]
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])#[4] 最终返回的proposals赋值给rpn_rois，作为rpn网络提供的建议区，注入后续FPN heads进行分类、目标框和像素分割的检测。 # [fan:] 第一阶段的提议输出，第二阶段Head的输入之一

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from. # 有的数据集中没有某个类别，计算后面的class_loss时，需要知道哪些类别是不存在的不应该计算，看了一个资料说，这份代码可以在多份数据中运行，需要统一类别吧 [fan] 涉及到active_class_ids相关如下，即将该图片隶属数据集中所有的class标记为1，不隶属本数据集合的class标记为0，计算Loss贡献时交叉熵会对每个框进行输出一个值，如果这个框最大的得分class并不属于其数据集，则不计本框Loss [3]
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)
            # active_class_ids: List of class_ids available in the dataset from which the image came.Useful if training on images from multiple datasets where not all classes are present in all datasets.

            if not config.USE_RPN_ROIS: # 这里的意思是可选择用RPN生成的提议框作为Head的输入，和不用RPN生成框而使用原始的数据集提供的目标框进行一定修改后作为Head的输入 [fan]
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else: # 一般情况，是用RPN生成的框训练Head [fan]
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            # [1] target_rois是第二部分ProposalLayer筛选提供的2000个区域建议框，在训练时，2000个显得太多，所以会进一步筛选为200个做为target。[4]
            # [fan] 如果要对Head进行训练，不仅需要对RPN生成的提议框下采样以减少数量(2000->200)，还需要我们自己生成每个RoIs的类别真值(应当属于哪个类别target_class_ids)、回归真值(与目标框的回归偏移量是多少target_bbox)、RoIs的掩模 , 原始数据提供的仅有RoIs的边界框(target_rois)、原图中物体的边界框(gt_boxes)、原图中物体的类别(input_gt_class_ids)、物体的掩模Mask(其个数是物体的个数 input_gt_masks)
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # [1] 分支1 fpn_classifier_graph用于分类和回归目标框偏移的
            # [1] 将rois在对应mrcnn_feature_maps特征层进行roialign特征提取，然后再经过各自的卷积操作预测最终结果。 [fan]mrcnn_class_logits 是RoIs的类别预测值(softmax前)，mrcnn_class 是RoIs的类别预测值(softmax后), mrcnn_bbox 是Head预测的RoIs的回归偏移量
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, # [1] config.POOL_SIZE是分类分支roialign使用的池化大小，取7
                                     config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE) # [1] config.FPN_CLASSIF_FC_LAYERS_SIZE是分类分支中全连接层尺寸，取1024
            # [1] 分支2 build_fpn_mask_graph 用于像素分割
            # [1] 将rois在对应mrcnn_feature_maps特征层进行roialign特征提取，然后再经过各自的卷积操作预测最终结果。
            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,# [1] config.MASK_POOL_SIZE是mask分支roialign使用的池化大小，取14
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            # [1] losses可分为两部分组成，
            # [1] 一是rpn网络的损失，包括rpn前景/背景分类损失rpn_class_loss和 rpn目标框回归损失rpn_bbox_loss；
            # [1] 类别使用的是 softmax_cross_entropy # [fan] RPN中anchor的分类损失
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits]) # [fan] input_rpn_match 是所有anchor的类别真值；rpn_class_logits 是所有anchor的类别预测值
            # [1] smooth_L1_loss # [fan] RPN中anchor的回归损失
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox]) # [fan] input_rpn_bbox是选中的anchor的回归真值；input_rpn_match是所有anchor的类别真值;rpn_bbox是所有anchor的预测回归值
            # [1] 二是mask_rcnn heads损失，包括分类损失class_loss、回归损失bbox_loss和像素分割损失mask_loss
            # [1] softmax_cross_entropy # [fan] mask_rcnn heads中RoIs的分类损失
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids]) # RoIs的类别真值(应当属于哪个类别target_class_ids) ; mrcnn_class_logits 是RoIs的类别预测值(softmax前) [fan] ；涉及到active_class_ids相关如下，即将该图片隶属数据集中所有的class标记为1，不隶属本数据集合的class标记为0，计算Loss贡献时交叉熵会对每个框进行输出一个值，如果这个框最大的得分class并不属于其数据集，则不计本框Loss [3]
            # [1] smooth_l1_loss # # [fan] mask_rcnn heads中RoIs的回归(偏移)损失
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox]) # [fan] RoIs的回归真值(RoIs与目标框的回归偏移量是多少target_bbox) ； RoIs的类别真值(应当属于哪个类别target_class_ids) ; mrcnn_bbox 是Head预测的RoIs的回归偏移量
            # [1] binary_cross_entropy # 对 target_mask 和 预测的mrcnn_mask中的target_class_ids的mask 求二分类交叉熵(单元是某类物体/不是某类物体)
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask]) # target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width] # [fan:] 每个RoIs的掩模 ; 每个RoIs的类别真值(应当属于哪个类别target_class_ids) ; pred_masks:预测的每个RoIs的每个类别的Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads # [fan] 测试时 和 训练时 喂入 Heads 的RoIs个数不一样，训练时从2000个RoIs 下采样到 200个，测试时是直接将1000个RoIs传给Heads。猜测：训练时2000个应该是偏多了，测试时考虑到召回率，没有下采样。
            # Proposal classifier and BBox regressor heads # [fan] Heads分类和回归预测输出 , 训练与测试 使用的函数/模型是一样的（但训练时分类回归分支 和 分割分支是并行的，测试时是先分类回归，筛选掉一些框后，最后做分割）
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates # [fan:] 根据Head预测的分类和回归输出，对1000个RoIs进行筛选，剔除掉RoIs中预测概率低的、预测为背景的、冗余高度重叠的，然后根据最高分类概率取top-k个RoIs的偏移量，RoIs微调后会作为Mask分支的输入
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections # [3] 我们已经获取了待检测图片的分类回归信息，我们将回归信息（即待检测目标的边框信息）单独提取出来，结合金字塔特征mrcnn_feature_maps，进行Mask生成工作（input_image_meta用于提取输入图片长宽，进行金字塔ROI处理，即PyramidROIAlign）。# [fan] 测试时，是求top-k个RoIs回归分支偏移后的框的Mask
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,#[3] 14
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)


            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')
            # [3] 整理一下模型输出Tensor：
            # num_anchors,    每张图片上生成的锚框数量
            # num_rois,       每张图片上由锚框筛选出的推荐区数量，
            # #               由 POST_NMS_ROIS_TRAINING 或 POST_NMS_ROIS_INFERENCE 规定
            # num_detections, 每张图片上最终检测输出框，
            # #               由 DETECTION_MAX_INSTANCES 规定

            # detections,     [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            # mrcnn_class,    [batch, num_rois, NUM_CLASSES] classifier probabilities
            # mrcnn_bbox,     [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            # mrcnn_mask,     [batch, num_detections(top-k), MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
            # rpn_rois,       [batch, num_rois, (y1, x1, y2, x2, class_id, score)]
            # rpn_class,      [batch, num_anchors, 2]
            # rpn_bbox        [batch, num_anchors, 4]

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)


        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.)) # [fan] config里面设置的都是1 ； get 字典中取键值，默认是1    https://www.runoob.com/python/att-dictionary-get.html
            self.keras_model.add_loss(loss) # 所有损失加起来

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model? 可能会有嵌套的model，给model递归设置trainable [fan]
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable? 根据正则表达式进行选择layer [fan]
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape) # 计算各个特征层shape [3]
        # Cache anchors and reuse if image shape is the same # 如果图片的形状确定了，那么anchor也是确定的，只用生成一次anchor存到内存里后可以复用 [fan]
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES, # anchor的面积 (32^2, 64^2, 128^2, 256^2, 512^2)[fan]
                self.config.RPN_ANCHOR_RATIOS, # anchor长宽比 [0.5, 1, 2] [fan]
                backbone_shapes, # 特征图长宽 256，128，64，32，16 [fan]
                self.config.BACKBONE_STRIDES, # [4:] BACKBONE_STRIDES 是特征图的降采样倍数，取[4, 8, 16, 32, 64]
                self.config.RPN_ANCHOR_STRIDE) # 是锚框采样的步长，取1 [4]
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a # 这一步是为了可视化，如果不需要可视化可以删除 [fan]
            # Normalize coordinates 归一化 [fan]
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)] # 返回对应图片形状的anchor [fan]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    # 将列表形式按顺序存储的数据属性，转换为字典形式的 [fan]
    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool) # [fan] 每个框的四个坐标加和得到一些值，对这些值中大于0的为True,小于或等于0的为False。这样就把坐标全零的框标识出来了。
    boxes = tf.boolean_mask(boxes, non_zeros, name=name) # [fan] 通过bool值进行筛选
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]]) # 这里使用从头切片而不是索引，是因为这里的真值 有意设置为先放正类，再放负类，最后放填充的无关类。
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape): # 将框的四个坐标值归一化 [fan]
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    # h，w 分别为输入图像的高和宽 [1]
    h, w = tf.split(tf.cast(shape, tf.float32), 2) # 1.第二个参数是整数2，这个整数代表这个张量最后会被切成2个小张量，shape:(2,)被平分成两个子tensor:h和w .  https://blog.csdn.net/SangrealLilith/article/details/80272346 [fan]
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0) # [1] 减1是因为图像以0为起点  [fan]  1.concat(): Concatenates tensors along one dimension. 2.Negative axis are interpreted as counting from the end of the rank  https://tensorflow.google.cn/versions/r1.12/api_docs/python/tf/concat
    shift = tf.constant([0., 0., 1., 1.]) # (y1, x1, y2, x2) [1] 为什么左上角坐标减0，而右下角减(1,1) [fan]
    return tf.divide(boxes - shift, scale) # 归一化到[0, 1]之间 [1] 这里是浮点数相除，所以可以保留小数点 [fan]


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
