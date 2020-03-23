# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
#@torchsnooper.snoop()
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torchsnooper
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')


def to_torch(ndarray):          
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)      #torch.from_numpy,该命令，将numpy数据转换为tensor
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):        #输入是cv2读入的单张图像h*w*c
    img = np.transpose(img, (2, 0, 1))  # C*H*W    yuantu:h*w*c
    img = to_torch(img).float()   #torch.float()将该tensor投射为float类型
    return img

#                           ,(cx,cy),127       ,s_z
@torchsnooper.snoop()
def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):  #pos就是指示的目标中心坐标（template-gt，search--上一帧预测的）
#     siamese_init中用法：z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
#     作用是由图像数据，(cx,cy),127,需要放大到的size：s_z得到（127*127*3）的template
#           由图像数据，(cx,cy),255,需要放大到的size：round(s_x)得到（255*255*3）的search
    if isinstance(pos, float):
        pos = [pos,pos]
    sz = original_sz   #s_z
    im_sz = im.shape   #(h*w*3)
    c = (original_sz + 1) / 2    #template--41
    #作用是截取一个sz*sz*的框出来
    context_xmin = round(pos[0] - c)        #cx-64
    context_xmax = context_xmin + sz - 1   #context_xmin+sz-1
    context_ymin = round(pos[1] - c)        #cy-64
    context_ymax = context_ymin + sz - 1   #context_ymin+sz-1
    
    left_pad = int(max(0., -context_xmin))   #正的，则不需要pad，负的，则需要补context_*m**大小的pad
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    #any() 函数用于判断给定的可迭代参数是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
    if any([top_pad, bottom_pad, left_pad, right_pad]):  #需要padding的话
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:  #不需要padding的话
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        
        
        #im_patch_original  crop from im; pad或者无pad之后要进行resize

    if not np.array_equal(model_sz, original_sz):   #model_sz, original_sz不一致则执行下面代码
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:    #model_sz, original_sz一致，，则执行下面代码
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch
    #返回的im_patch的shape为(3,h,w)

#对Anchors类产生的chanors(anchor_num,4)再操作----->(anchor_num*score_size*score_size,4)
@torchsnooper.snoop()  
def generate_anchor(cfg, score_size):   #默认是25  #cfg是test.py中model.anchors字典
    anchors = Anchors(cfg)   #实例化Anchors会自动调用函数Anchors类里面的函数generate_anchors()来生成self.anchors
    anchor = anchors.anchors    #anchor =(anchor_num,4)   (x上,y上,x下,y下)  对应（x1, y1, x2, y2）
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)   #(anchor_num,4)  (cx,cy,w,h)

    total_stride = anchors.stride   #8
    anchor_num = anchor.shape[0]


    #按原来的方式广播得到所有的锚点。复制锚点，然后添加不同位置的偏移量。
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4)) #anchor横向扩大score_size*score_size倍
    #anchor.shape = (anchor_num*score_size*score_size,4),每一个anchor有shape为(score_size*score_size,4)的数据，
    #一共anchor_num个anchor数据竖着摞在一起，
    
    
    ori = - (score_size // 2) * total_stride  #ori = -96              #0-24
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],     #[-96,-88,80,....,88,96]
                         [ori + total_stride * dy for dy in range(score_size)])  #生成网格点坐标矩阵。
    #xx.shape = (score_size,score_size),yy.shape = (score_size,score_size)其实就是（25，25）、
    
    
    #把xx展平，然后横向扩大anchor_num倍，yy亦如此
    #此操作之后，xx.shape = (anchor_num*score_size*score_size,),yy.shape = (anchor_num*score_size*score_size,)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()   #np.tile,对数组进行重复操作
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor   #(anchor_num*score_size*score_size,4)



#siamese_init 构造state字典。输入的im是单张图像(h*w*3)，矩形中心(cx,cy)和(w,h)
@torchsnooper.snoop()
def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):#target_pos, target_sz输入的就是由gt轴对称得来的
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()        #配置参数
    p.update(hp, model.anchors)    #用hp,model.anchors更新p的参数，相当于用config_vot.json更新p
    p.renew()
#    p.score_size=25
    net = model
    p.scales = model.anchors['scales']  #Custom的父类SiamMask里的属性
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num   #vot数据集上是5
    
    
    p.anchor = generate_anchor(model.anchors, p.score_size)  #generate_anchor 生成锚点。p.anchor.shape = (p.anchor_num*p.score_size*p.score_size,4)
    avg_chans = np.mean(im, axis=(0, 1))   #此处im单张图片，对每个颜色通道都求均值（3，）（B,G,R)

    #图像预处理，按比例外扩目标框，从而获得一定的 context 信息。p.context_amount = 0.5
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)  #wc_z = w + p.context_amuont * (w+h)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)  #hc_z = h + p.context_amuont * (w+h)
    #需要将框定的框做一个大约2倍放大,以物体为中心， s_z为宽高，截取一个正方体的物体出来
    s_z = round(np.sqrt(wc_z * hc_z))  #round四舍五入取整，round(2.5) = 2,round(2.51)=3
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)#tensor<(3, 127, 127), float32, cpu>
                                          #TrackerConfig中的定义是 input z size，127
    #z_crop的维度是(127*127*3)
                                          
    z = Variable(z_crop.unsqueeze(0)) #pytorch中的命令，扩充数据维度,变成神经网络的参数tensor<(1, 3, 127, 127), float32, cpu>
    net.template(z.to(device))   #将z送到cuda上面提取特征，即得到resnet50之后的结果

    if p.windowing == 'cosine':   #默认
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  #求外积 ndarray(p.score_size，p.score_size)即<(25, 25), float64>
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
   
    window = np.tile(window.flatten(), p.anchor_num)   #对window.flatten()在X轴进行重复p.anchor_num次
    #ndarray<(3125,), float64>,p.anchor_num=5
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos   #还是传进来的数据
    state['target_sz'] = target_sz     #还是传进来的数据
    return state


    #state是siamese_init返回的结果，im是单张图像数据
    #作用：每一帧图像框定一个框,输出的state[target_pos,target_sz]和传入的不一样
@torchsnooper.snoop()
def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    p = state['p']     #state['p']是TrackerConfig类
    net = state['net'] #siamese_init中的model，模型即是网络
    avg_chans = state['avg_chans']   
    window = state['window']
    target_pos = state['target_pos']  #中心点坐标
    target_sz = state['target_sz']    #(w,h)
 
    #由扩展后的宽高计算等效面积。使用与模板分支相同的缩放系数得到检测区域。
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)  #h +p.context_amount*(w+h)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)  #w +p.context_amount*(w+h)
    s_x = np.sqrt(wc_x * hc_x)   #这个s_x是作为template时，以物体为中心，s_x为宽高，截取一个正方体的物体出来，然后再resize到（127，127），这个s_x时框的大约2倍的放大
    scale_x = p.exemplar_size / s_x     #scale_x是放大的倍数
    d_search = (p.instance_size - p.exemplar_size) / 2   #64
    pad = d_search / scale_x       #pad = 64*s_x/127
    s_x = s_x + 2 * pad       #这个s_x是作为search时，以物体为中心，s_x为宽高，截取一个正方体的物体出来，然后再resize到（255，255），这个s_x时框的大约4倍的放大
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]##(x上,y上,w,h)
    #crop_box就是search未resize的原图

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)  #np.int0向下取整
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)
    #提取按比例缩放的剪裁在之前的目标位置，为x
    # extract scaled crops for search region x at previous target position以上一帧的target_pos为依据生成search，毕竟物体位置差别不大。
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    #x_corp.shape = [255,255,3]----->(1, 3, 255, 255), float32, cpu>
    
    #运行网络
    if mask_enable:   #如果运训mask分支的话
        score, delta, mask = net.track_mask(x_crop.to(device))  #三分支分别的结果
#        New var:....... score = tensor<(1, 10, 25, 25), float32, cuda:0, grad>   2*k = 10
#        New var:....... delta = tensor<(1, 20, 25, 25), float32, cuda:0, grad>   4*k = 20  k = 5
#        New var:....... mask = tensor<(1, 3969, 25, 25), float32, cuda:0, grad>
    else:
        score, delta = net.track(x_crop.to(device))


   #解码出预测框，并根据位置、宽高比和位移量惩罚得分，挑选出最优预测。torch.permute(dims),将tensor的维度换位。
   #即使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形。
   #.data.cpu().numpy()  GPUtensor-->CPUtensor-->numpy
   #.data[:,1],取tensor所有行的第二列
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()  #ndarray<(4, 3125), float32>
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()    #score = ndarray<(3125,), float32>，，torch.nn.functional.softmax(input)非线性激活函数

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]   #cx = cx *w+cx
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]   #cy = cy * h+cy 
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]            #w = exp(w) *w
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]            #h = exp(h) * h
    #np.maximum：(X, Y, out=None)
    #X 与 Y 逐位比较取其大者；
    def change(r):
        return np.maximum(r, 1. / r)   #[0.33, 0.5, 1, 2, 3],[3.03,2,1, 0.5,0.333] --->[3, 2, 1, 2, 3]

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz*scale_x   #target_sz_in_crop = ndarray<(2,), float64>
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty   ndarray<(3125,), float32>
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty   ndarray<(3125,), float32>

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k) #ndarray<(3125,), float32>
    pscore = penalty * score                        #ndarray<(3125,), float32>

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence  #ndarray<(3125,), float64>
    best_pscore_id = np.argmax(pscore)    #挑选出得分最高的。通过class score分支的最高得分选取一根柱子用于生成mask，同时也将对应的最优框选出来了

    pred_in_crop = delta[:, best_pscore_id] / scale_x   #找到在search中偏差的位置，用于选择最优框pred_in_crop = ndarray<(4,), float32>，其实是偏差
    
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]   #x的偏差加上原来的x
    res_y = pred_in_crop[1] + target_pos[1]   #y的偏差加上原来的y

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])   #从这里往下，target_pos和target_sz改变。
    target_sz = np.array([res_w, res_h])    #解码出来的框，就一个

    # for Mask Branch
    #numpy.unravel_index 将平面索引或平面索引数组转换为坐标数组的元组。
    #由best_pscore_id得到特征图上的torchsnooper位置。track_refine 函数运行 Refine 模块，
    #由相关特征图上 1×1×256的特征向量与检测下采样前的特征图得到目标掩膜。


#上面的mask是整体的，现在要得出一根ROW
    if mask_enable:    #允许mask
       
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))  #(3,delta_y,delta_x) 在pscore中找对应的mask中的位置
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
        #根据最高得分id选择一根最优柱子用于生成mask，现在已经有坐标位置了（delta_y, delta_x）
        if refine_enable:   #用sharprefine模块
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()   #mask = <tensor<(1, 16129)------>darray<(127, 127), float32>
        else:       #不用sharprefine模块，而是选出最高得分的mask
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()    #ndarray<(127, 127), float32>，因为p用model.anchors更新了一下
                
                
        #上面生成了mask,现在要映射回原图，warpAffine() 对图像应用仿射变换。
        #手动构造变换矩阵mapping，a和b为尺度系数，c和d为平移量。
        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop
 
        #crop_box为检测截取框，格式为[x,y,width,height]。s为缩放系数。sub_box为预测的模板区域框。
        s = crop_box[2] / p.instance_size        #s是标量，，round(s_x)/255
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,   #x上 + (delta_x - 4) * 8 * s
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,   #y上 + (delta_y - 4) * 8 * s
                   s * p.exemplar_size, s * p.exemplar_size]     #列表，四个元素      #s*127,s*127
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]   #列表，四个元素
        
        #输入的mask为ndarray<(127, 127), float32>
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))   #和im原视频帧size一样,,float32
        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)   ##和im原视频帧size一样,uint8>二维 0-1值
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:   #这个成立我的版本是3.*
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #三维，就一个轮廓
            
            
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:   #有轮廓并且最大的轮廓面积大于100，说明轮廓不小
            contour = contours[np.argmax(cnt_area)]  # use max area polygon，，(n , 1 , 2)维的numpy.ndarray，n是坐标的个数
            polygon = contour.reshape(-1, 2)          #二维
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated RectangleboxPoints 查找旋转矩形的四个顶点。用于绘制旋转的矩形。
            #cv2.minAreaRect(polygon)，生成最小外接矩形，输入时多边形点集必须是array数组的形式，输出是（中心(x,y), (宽,高), 旋转角度）
            #cv2.boxPoints(rect)获取最小外接矩形的4个顶点坐标,返回形式[ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]。
            #prbox = ndarray<(4, 2)，就是得到的目标框的四个顶点，俗称旋转框，因为有角度
                
            # box_in_img = pbox
            rbox_in_img = prbox    #rbox_in_img = ndarray<(4, 2), float32>
        else:  # empty mask轮廓太小的话
            location = cxy_wh_2_rect(target_pos, target_sz)    #得到左上角坐标（x，y)和(w,h)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
            
#    到此，rbox_in_img就是经过“和原图size一样大的mask”经过轮廓操作或者直接由预测的(target_pos, target_sz)生成的--目标框的四个顶点
            
            
    #由结果更新状态。
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state
#video是单个视频构成的字典
@torchsnooper.snoop()
def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False, device='cpu'):
    #regions记录目标框以及状态。
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']   #gt是groundtruth(325, 8)，数组,img_files是一个视频中所有路径构成的列表
    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0   #len(image_files)是视频帧数，每个视频不同
    
    #遍历当前视频中的所有图像，f是索引，image_file是单个帧的路径，由目标出现的帧初始化。
    for f, image_file in enumerate(image_files):    
        im = cv2.imread(image_file)      #(h*w*3)
        tic = cv2.getTickCount()     #记录当前时间
        if f == start_frame:  # init初始化，如果是第一帧
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])  #将gt中任意方向的矩形（标识目标）转换成轴对称的矩形
            target_pos = np.array([cx, cy])    #轴对称矩形中心
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, model, hp, device)  # init tracker初始化跟踪器
                    #输入是一帧图像数据，gt的(cx,cy)和(w,h), model, hp, device
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])   #得到的location为轴对称矩形左上角坐标(x,y,w,h)ndarray<(4,), float64>
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking ，接下来的所有帧,在后续帧跟踪，由state获取目标框。
            if f ==3:
                exit()
            state = siamese_track(state, im, mask_enable, refine_enable, device, args.debug)  # track
            if mask_enable:
                location = state['ploygon'].flatten()
                mask = state['mask']   #mask.shape和原图一样
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])#解码出的预测框左上角(x,y),(w,h)
                mask = []

            if 'VOT' in args.dataset:
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                if mask_enable:
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                else:
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
#                无论怎样locatioon都是预测得到的，计算预测和实际的多边形之间的重叠
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:           #vot_overlap 计算两个多边形gt_polygon, pred_polygon之间的重叠。
                b_overlap = 1

            if b_overlap:   #值为真，有重叠
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        #处理完one video后进行显示 
        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, str(state['score']) if 'score' in state else '', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    
#    下面是后续处理工作
    # save result跟踪完成，记录结果到文本文件。
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') +\
           args.resume.split('/')[-1].split('.')[0]

    if 'VOT' in args.dataset:
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                        fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    else:  # OTB
        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


    #获得预测掩膜及标注的数量。构造预测结果的 id 列表object_ids。
def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    #output_max_id为每个像素位置预测的 id，0通道为背景。
    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res

#video是单个视频构成的字典
@torchsnooper.snoop()
def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu'):
    image_files = video['image_files']

#PIL.Image.open()专接图片路径，用来直接读取该路径指向的图片。分割数据的标注亦为图片
    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    if 'anno_init_files' in video:   #如果标注初始文件在video里
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    #"DAVIS2017"和"ytb_vos"会开启多目标跟踪。
    if not mot_enable:   #mot_enable为False则执行下面语句
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:   #'start_frame' 不在video这个字典中
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)  #object_num是多目标跟踪的目标数目
    
   #每个目标都遍历图像，在起止帧之间执行跟踪。pred_masks记录所有的 mask。 
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))-1
    for obj_id, o_id in enumerate(object_ids):   #每个目标对单个视频

        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):   #一个视频的每个图像
            im = cv2.imread(image_file)
            tic = cv2.getTickCount()        #计算起始时间
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                # boundingRect() 计算点集或灰度图像的非零像素的垂直矩形。np.uint8，也就是0~255
                x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])   #矩形中心坐标
                target_sz = np.array([w, h])      #size
                state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker
            elif end_frame >= f > start_frame:  # tracking在第一帧到最后一帧之间
                state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
                mask = state['mask']
            toc += cv2.getTickCount() - tic  #计算的是两次大循环的总时间
            if end_frame >= f >= start_frame:  #对每一帧都要预测mask
                pred_masks[obj_id, f, :, :] = mask
    toc /= cv2.getTickFrequency()


    #MultiBatchIouMeter 批量计算 IoU
    if len(annos) == len(image_files):
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                           multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []

    #pred_mask_final合并图像上多个目标的模板索引，默认0通道为背景。索引直接保存为图片无法可视化。
    if args.save_mask:
        video_path = join('test', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            cv2.imwrite(join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png'), pred_mask_final[i].astype(np.uint8))

    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        for f, image_file in enumerate(image_files):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f,:,:,:])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f*len(object_ids) / toc))

    return multi_mean_iou, f*len(object_ids) / toc

def main():
    global args, logger, v_id    #全局变量
    args = parser.parse_args()   #args是test.py文件运行时，接受的参数
    cfg = load_config(args)      #加载 JSON 配置文件并设置args.arch的值。
    print(cfg)

    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)#add_file_handler 创建一个记录器并绑定文件句柄。

    logger = logging.getLogger('global')
    logger.info(args)

    # setup model         Custom 为论文实现的网络。如果不是“Custom”，加载 models 下指定的结构。
    if args.arch == 'Custom':    #args.arch参数，预训练模型的结构，命令行不给的话，默认为' ',
        from custom import Custom
        model = Custom(anchors=cfg['anchors']) #cfg是从config_vot.json的到的数据，所以跟踪时用的model.anchors字典中的数据
    else:
        parser.error('invalid architecture: {}'.format(args.arch))

    if args.resume:    #给了args.resume,如果args.resume不是文件，报错，
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)  #args.resume是文件load_pretrain ,能够处理网络之间的不一致
    model.eval()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)
    
    # setup dataset，字典
    dataset = load_dataset(args.dataset)  #load_dataset 能够加载 VOT、DAVIS、ytb_vos 三种数据集。
                                            #仅以上三种数据源支持掩膜输出。
                                            
    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output  ,使用掩膜输出
    else:
        vos_enable = False

    total_lost = 0  # VOT  跟踪任务有损失函数
    iou_lists = []  # VOS  分割任务
    speed_list = []

    #v_id视频索引从1起,video是视频名字
    for v_id, video in enumerate(dataset.keys(), start=1):   
        if v_id ==2:
            exit()
        if args.video != '' and video != args.video:   #不成立，args.video默认是' '
            continue

        if vos_enable:  #分割任务,,,,分割任务和跟踪任务只能选一个
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], device=device)
            iou_lists.append(iou_list) #iou_list是什么类型的数据？？？
        else:          #跟踪任务
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                             args.mask, args.refine, device=device)
            total_lost += lost
        speed_list.append(speed)

    # report final result记录最终结果
    if vos_enable:  #如果进行的是分割任务
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))

if __name__ == '__main__':
    main()
