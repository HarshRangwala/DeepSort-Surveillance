from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import insightface
from insightface.utils import face_align

from sklearn import preprocessing

def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, prefix, epoch, layer):
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, ctx_id, model_prefix, model_epoch, use_large_detector=False):
        if use_large_detector:
            self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        else:
            self.detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
        self.detector.prepare(ctx_id=ctx_id)
        if ctx_id>=0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (112,112)
        self.model = get_model(ctx, image_size, model_prefix, model_epoch, 'fc1')
        self.image_size = image_size

    def get_input(self, face_img):
        bbox, pts5 = self.detector.detect(face_img, threshold=0.8)
        if bbox.shape[0]==0:
            return None
        bbox = bbox[0, 0:4]
        pts5 = pts5[0, :]
        nimg = face_align.norm_crop(face_img, pts5)
        return nimg
    def get_input_all(self, face_img):
        bbox, pts5 = self.detector.detect(face_img, threshold=0.5)
        if bbox.shape[0]==0:
            return None,None,None,None
        if bbox is None:
            return None, None, None, None
        output=[]
        faces_=[]
        key_points_=[]
        bboxes_=[]
        for face,point in zip(bbox,pts5):
            bbox = face[0:4].astype(np.int)
            #return(bbox)
            # print (point.shape,point,bbox.shape,bbox)
            #point=point.reshape((2,5)).T
            to_add_face=face_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            faces_.append(to_add_face)
            key_points_.append((point.astype(np.int),face[4]))
            bboxes_.append(bbox)
            # cv2.imshow("cropped",to_add_face)
            nimg = face_align.norm_crop(face_img, point)
            # cv2.imshow("alligned_cropped",nimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2,0,1))
            output.append(aligned)
        return np.array(output),faces_,np.array(key_points_),np.array(bboxes_)

    def get_feature(self, aligned):
        a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()[0]
        norm = np.sqrt(np.sum(emb*emb)+0.00001)
        emb /= norm
        return emb
    def get_batch_feature(self, aligned):
        input_blob = aligned
        #print(input_blob.shape)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding)
        return embedding

