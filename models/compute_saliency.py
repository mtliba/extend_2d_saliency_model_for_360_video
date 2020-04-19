
import os 
from Salicon import main
from _3DSal import test
from TASEDNET import TASEDNET 
from ACLNet import _main
from SalEMA import inference

def compute_sal(model  ,header ,output ,project =True):
    # we need to know if it s projected or not because it s loading images not video
    if model == 'Salicon' :
        if project :
            for vid in os.listdir(header) :
                if not os.path.isdir(os.path.join(header)) :
                        continue
                main.call(os.path.join(header ,vid),output)
        else :       

            main.call(os.path.join(header),output)

    # header is set video path
    if model == 'SalEMA' :
        
        inference.call(header,output)
    if model == 'TASEDNET' :
        
        TASEDNET.call(header,output)
    if model == 'ACLNet' :
        
        _main.call(header,output)  
    if model == '3DSal' :
        
        test.call(header,output)     
        
