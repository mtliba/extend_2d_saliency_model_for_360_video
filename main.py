import cv2
import os
import datetime
import numpy as np
from models.compute_saliency import compute_sal
from args import get_inference_parser
from project import project_to_cmp ,project_to_erp
from scipy.misc.pilutil import imsave , imread
from PIL.Image import open , save 
from merge import sh_merge
from ERP2CMP import ERP_2CMP
from CMP2ERP import CMP_2ERP

def main(args):

        # get arguments 
        model  = args.model
        src    = args.src
        dst    = args.dst
        method = args.method
        m_tech = args.tech
        to_bin = args.bin
        
        # exit if model not recognized 
        if not model in ['TASED' ,'SalEMA' ,'ACLNet' ,'Salicon' ,'3DSal'] :
            print("Your model was not recognized, check the name of the model and try again.")
            exit()  

        # exit if test folder is empty
        if not os.path.exists(src) or len(os.listdir('./'+src)) == 0 :
            print(' you directory is enpty')
            exit()
        
        # create output folder if it does not exist 
        if not os.path.exists(dst) :
            os.mkdir('./'+dst)
        
        # if CMP or MERGED we need to apply CMP projection and calcul saliency on cube faces 
        if method !='ERP' :
            # projected dataset
            if not os.path.exists('./CMP') :
                os.mkdir('./CMP')
            # resulted saliency prediction on projected dataset
            if not os.path.exists('./CMP_sal') :
                os.mkdir('./CMP_sal')
            
            # invers projected dataset 
            if not os.path.exists('./ERP_after_Project') :
                os.mkdir('./ERP_after_Project')
            
            list_of_videos = os.listdir(src)

            # apply CMP on the whole dataset
            ERP_2CMP(src ,'./CMP')

            # each video will result 6 video owr new subdataset  
            for vid in list_of_videos :

                if not os.path.isdir(os.path.join(src,vid)) :
                    continue
                CMP_header = './CMP/'+vid
                output= './CMP_sal'+vid
                # calcul dynamic saliency over each cube face ,each cube face consider as 2d video 
                compute_sal(model = model ,header = CMP_header  ,output= './CMP_sal'+vid)
                # invers project saliency result of each subdataset
                CMP_2ERP(output,'./ERP_after_Project')

        # method is ERP or MERGED
        # in case of method is ERP or MERGED we need to calcul result of ERP
        if method =='ERP' or method =='MERGED' :

            # if method is ERP save in dst putted by user
            if method =='ERP' :
                new_dst =  dst
            else :
                if not os.path.exists('./ERP') :
                    os.mkdir('./ERP')
                new_dst = './ERP'

            list_of_videos = os.listdir(src)
            
            for vid in list_of_videos :
                if not os.path.isdir(os.path.join(src,vid)) :
                    continue

                if not os.path.exists(os.path.join(new_dst ,vid)) :
                    os.mkdir(os.path.join(new_dst ,vid))


                ERP_images_header = os.path.join(src,vid)
                # compute saliency map ensure that it has the same size
                compute_sal(model = model ,header = ERP_images_header ,output= os.path.join(new_dst ,vid))
                

        # apply merging between both map
        if method == 'MERGED' :
            list_of_videos = os.listdir(src)

            if m_tech == 'spherical' :
                for vid in list_of_videos :
                    if not os.path.isdir(os.path.join(src,vid)) :
                        continue

                    lsit_of_frames = [img for img in os.listdir(os.path.join(src,vid)) if img.endswith(".png")] 
                    lsit_of_frames.sort() 
                    # merged is true mean ' 
                    for img in lsit_of_frames :
                        merged_img = sh_merge(ERP_after_project = os.path.join('./ERP_after_Project' ,vid ,lsit_of_frames[i]), ERP =os.path.join('./ERP' ,vid ,lsit_of_frames[i]))
                        shift_content = True 
                        if shift_content  :
                            pass
                        # save merged image 
                        merged_img.save( os.path.join(dst,vid ,lsit_of_frames[i]))
                    
                        # write video saliency value in bin file  
                        if to_bin == True :

                            os.mkdir('./bin')
                            os.chdir('./bin/')
                            with open(str(vid)+'_2048x1024x'+str(len(lsit_of_frames))+'_32b.bin', 'ab') as the_file:
                                for i in range(merged_img.shape[0]):
                                    if i in range(50) or i in range(973,merged_img.shape[0]):
                                        line = np.float32(np.zeros(merged_img.shape[1]))
                                    else : 
                                        line = np.float32(np.array(merged_img[i]))
                                    the_file.write(bytes(line))                             
                    
            # spatial merge
            elif m_tech == 'spatial' :
                pass

            # merge based on a polynome 
            elif m_tech == 'serie' :
                pass
            
            # add some print about time ...

if __name__ == '__main__':
    
    # parse read argument 
    
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args)    