import argparse


def get_inference_parser():

    parser = argparse.ArgumentParser(description="Video saliency prediction framework over 360 scene  . Use this script to get saliency maps for your videos")
    # gpu support
    parser.add_argument('-use_gpu', dest='use_gpu', default=True, type=bool, help="Boolean value, set to True if you are using CUDA.")
    # pths
    parser.add_argument('-dst', dest='dst', default="./output", help="Add root path to output predictions to.")

    parser.add_argument('-src', dest='src', default="./exemple", help="Add root path to your dataset.")

    # Model
    parser.add_argument('-model', dest='model', default='TASEDNet', help="Input your desired model name 'TASEDNet ,SalEMA ,ACLNET ,Salicon ,3DSal'")
    
    # Projection 
    parser.add_argument('-project', dest='method', default='MERGED', help="Input your desired projection method 'ERP ,CMP ,MERGED'")
    
    # merge technique 
    parser.add_argument('-merge', dest='tech', default='shperical', help="Input your desired merge technique 'shperical ,spatial ,serie'")
    
    # create bin file
    parser.add_argument('-bin', dest='bin', default=False, type=bool, help="crate bin file contain saliency value")
   
    
    return parser


if __name__ =="__main__":

    parser = get_inference_parser()
    args_dict = parser.parse_args()
    print(args_dict)