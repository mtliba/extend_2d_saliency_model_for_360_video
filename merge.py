from projection import methods
import numpy as np
from PIL import Image
import os
import cv2

def sh_merge(ERP_after_project , ERP):
    
    ERP_after_project = np.array(ERP_after_project)
    if len(ERP_after_project.shape) == 2 :
        ERP_after_project = ERP_after_project[..., None]
    
    ERP = np.array(ERP)
    if len(ERP.shape) == 2 :
        ERP = ERP[..., None]
    if ERP_after_project.shape != (500,1000) : ERP_after_project = cv2.resize(ERP_after_project,(1000,500))
    
    if ERP.shape != (500,1000) : ERP = cv2.resize(ERP,(1000,500))
    
    out_D = methods.e2c(ERP_after_project)
    out_A = methods.e2c(ERP)
    fin={}
    for face_key in out_D:

        out_A[face_key] = cv2.cvtColor(out_A[face_key], cv2.COLOR_BGR2GRAY)
        out_D[face_key] = cv2.cvtColor(out_D[face_key], cv2.COLOR_BGR2GRAY)

        out_D[face_key] = out_D[face_key].reshape((256,256))
        
        out_A[face_key] = out_A[face_key].reshape((256,256))

        if face_key == 'U' or face_key == 'D':
            out = 0.2*out_D[face_key] +0.8*out_A[face_key]
            out_N = out_A[face_key]
            out[out < 0.1*out_N] = 0.2*out_N[out < 0.1*out_N]              
        else :
            out = 0.8*out_D[face_key] + 0.2*out_A[face_key]
            out_N = out_D[face_key]
            out_M = out_A[face_key]
            out = out_M
            out[out_M < out_N] = 0.8*out_N[out_M < out_N]

        fin[face_key] =cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    fin_e = methods.c2e(fin, h=1024, w=2048, mode='bilinear',cube_format='dict')
    
    return fin_e

    