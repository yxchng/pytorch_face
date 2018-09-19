import time
import argparse
from tqdm import tqdm
import torch

import pytorch_face.models as models
from pytorch_face.transforms import *
from pytorch_face.evaluations import *
from pytorch_face.utils import *

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--mean', default=[127.5, 127.5, 127.5], type=float, nargs="+", metavar='PARAM',
                    help='mean')
parser.add_argument('--std', default=[128, 128, 128], type=float, nargs="+", metavar='PARAM',
                    help='standard deviation')
parser.add_argument('--lfw_dir', metavar='PATH', default='./data/test/lfw-112x96', 
                    help='path to test data')
parser.add_argument('--ijbc_dir', metavar='PATH', default='./data/test/ijbc-112x96', 
                    help='path to test data')
parser.add_argument('--metadata_dir', metavar='PATH', default='./metadata/test', 
                    help='path to metadata')
parser.add_argument('--model_dir', type=str,
                    help='path to models directory')
parser.add_argument('--model_ext', type=str, default=".pth.tar",
                    help='path to models directory')
args = parser.parse_args()

def main():
    print("==> parsing testing configurations")
    for arg in vars(args):
        print(' - {0:<16}: {1}'.format(arg, getattr(args, arg)))
    device =  torch.device('cuda')


    trans = transforms.Compose([
        transforms.Normalize(args.mean, args.std),
        transforms.ToTensor(),
        transforms.ToDevice(device)
    ])

    for root, _, file_names in os.walk(args.model_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if file_path.endswith(args.model_ext):
                checkpoint = torch.load(file_path)
                title = "| {0} |".format("TESTING MODEL '{0}'".format(file_name))
                print("+{0}+".format("="*(len(title)-2)))
                print(title)
                print("+{0}+".format("="*(len(title)-2)))
                print(" - {0:<18}: {1}".format("arch", checkpoint['arch']))
                print(" - {0:<18}: {1}".format("margin_type", checkpoint['metamodel']['margin_type']))
                print(" - {0:<18}: {1}".format("margin_parameters", checkpoint['metamodel']['margin_parameters']))
                print(" - {0:<18}: {1}".format("epoch", checkpoint['epoch']))
                print(" - {0:<18}: {1}".format("loss", checkpoint['loss']))
                print(" - {0:<18}: {1}".format("prec1", checkpoint['prec1'].item()))
                print(" - {0:<18}: {1}".format("prec5", checkpoint['prec5'].item()))
                model = models.__dict__[checkpoint['arch']](num_classes=checkpoint['metamodel']['num_classes'],
                                                            margin_type=checkpoint['metamodel']['margin_type'],
                                                            margin_parameters=checkpoint['metamodel']['margin_parameters'],
                                                            is_deploy=True)   
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)

                # lfw
                #if args.lfw_dir:
                    #fold_thresholds, fold_scores = lfw.evaluate_verification(model, trans, args.lfw_dir, args.metadata_dir)

                # ijbc
                if args.ijbc_dir:
                    # verification
                    far_levels = [0.00001, 0.0001, 0.001, 0.01, 0.1]
                    #tar_at_far = ijbc.evaluate_verification(model, trans, args.ijbc_dir, args.metadata_dir, far_levels=far_levels, agg_func=np.mean, normalize_func=ijbc.normalize, pool_func=None)
                    #tar_at_far = ijbc.evaluate_verification(model, trans, args.ijbc_dir, args.metadata_dir, far_levels=far_levels, agg_func=np.max, normalize_func=ijbc.normalize, pool_func=None)
                    #tar_at_far = ijbc.evaluate_verification(model, trans, args.ijbc_dir, args.metadata_dir, far_levels=far_levels, agg_func=None, normalize_func=None, pool_func=ijbc.average_then_normalize)
                    #tar_at_far = ijbc.evaluate_verification(model, trans, args.ijbc_dir, args.metadata_dir, far_levels=far_levels, agg_func=None, normalize_func=None, pool_func=ijbc.normalize_then_average)

                    # identification
                    rank_levels = [1, 5, 10, 20]
                    fpir_levels = [0.0001, 0.001, 0.01, 0.1]
                    cmc = ijbc.evaluate_identification(model, trans, args.ijbc_dir, args.metadata_dir, rank_levels=rank_levels, fpir_levels=fpir_levels, 
                                                       normalize_func=None, pool_func=ijbc.average_then_normalize, agg_func=None)
                    cmc = ijbc.evaluate_identification(model, trans, args.ijbc_dir, args.metadata_dir, rank_levels=rank_levels, fpir_levels=fpir_levels, 
                                                       normalize_func=None, pool_func=ijbc.normalize_then_average, agg_func=None)
                    #cmc = ijbc.evaluate_identification(model, trans, args.ijbc_dir, args.metadata_dir, rank_levels=rank_levels, fpir_levels=fpir_levels, 
                    #                                   normalize_func=ijbc.normalize, pool_func=None, agg_func=np.mean)
                    #cmc = ijbc.evaluate_identification(model, trans, args.ijbc_dir, args.metadata_dir, rank_levels=rank_levels, fpir_levels=fpir_levels, 
                    #                                   normalize_func=ijbc.normalize, pool_func=None, agg_func=np.max)
                

if __name__ == '__main__':
    main()

