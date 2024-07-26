import pytorch101
import torch

def main():
    print("pytroch版本: " + torch.__version__)
    print("cuda是否可用: " + str(torch.cuda.is_available()))   
    print("查看cuda版本: "+ str(torch.version.cuda))          
    print("cudnn是否可用: " + str(torch.backends.cudnn.is_available()))
    print("cudnn版本: " + str(torch.backends.cudnn.version()))   
    
if __name__ == '__main__':
    main()
    