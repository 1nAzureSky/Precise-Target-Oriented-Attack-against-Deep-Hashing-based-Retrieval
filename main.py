import os.path
from evaluate import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='data', default='NUS-WIDE',choices=['FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],help='dataset')
parser.add_argument('--data_path', dest='data_path', default='',help='dataset path')
parser.add_argument('--hashing_backbone', dest='h', default='CSQ', help='Hashing method')
parser.add_argument('--network_backbone', dest='net', default='ResNet50', help='Backbone network')
parser.add_argument('--target_model', dest='target_model', default='ResNet18', help='target network')

parser.add_argument('--bit', dest='bit', type=int, default=32, choices=[16,32,64],help='Hash code length')
parser.add_argument('--attack', dest='attack', type=bool, default=False,choices=[True,False], help='in/out-of-classes case')
parser.add_argument('--replace', dest='replace', type=bool, default=False,choices=[True,False], help='set True when testing general target label selection')

parser.add_argument('--multi', dest='multi', type=bool, default=False,choices=[True,False], help='general-target label selection')
parser.add_argument('--transfer', dest='transfer', type=bool, default=False,choices=[True,False], help='black-box')

parser.add_argument('--pos_size', dest='pos_size', type=int, default=75, help='Numbers of positive sample')
parser.add_argument('--iteration', dest='iteration', type=int, default=20, help='optimation iteration')
parser.add_argument('--cuda', dest='cuda', type=int, default=0, help='CUDA devices index')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255, help='epsilon')
parser.add_argument('--eta', dest='eta', type=float, default=0.1, help='eta')

parser.add_argument('--model_path', dest='model_path', default='./model_weights/', help='')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'
random.seed(3)

args.data_path=f'{args.data_path}{args.data}'
args.model_path = f'{args.model_path}{args.data}_{args.h}_{args.net}_{args.bit}.pth'
if not args.transfer:
    args.database_hash_path = f'./log/database_code_{args.data}_{args.h}_{args.net}_{args.bit}.txt'
else:
    args.database_hash_path = f'./log/database_code_{args.data}_{args.h}_{args.target_model}_{args.bit}.txt'
print(args.database_hash_path)

args.target_label_path = f'./log/target_label_{args.data}_{str(args.attack)}.txt' if not args.multi else f'./log/target_label_{args.data}_multi.txt'
args.txt_path = f'./data/{args.data}'
args.save_pr = f'./logs/{args.data}_{args.h}_{args.net}_{args.bit}.json'
args.target = 1 if args.attack else 0
args.attack_case = 'Attacking in-classes label' if args.attack else 'Attacking out-of-classes label'
if args.data == 'NUS-WIDE':
    args.num_cls = 21
    args.topk = 5000
elif args.data == 'FLICKR-25K':
    args.num_cls = 38
    args.topk = 5000
elif args.data == 'MS-COCO':
    args.num_cls = 80
    args.topk = 5000

args.model = load_model(args.model_path)

print(f'Data : {args.data}\nHashing Method : {args.h}\nBackbone : {args.net}\nBit : {args.bit}\nUsing cuda : {args.cuda}\nParameter setting : Pos_size : {args.pos_size}, Iteration : {args.iteration}\n{args.attack_case}')
if args.multi:
    map,map_clean_ori,map_clean,map_clean_ori_multi = evaluate_multi(args)
else:
    map,map_clean_ori,map_clean,map_clean_ori_multi = evaluate(args)
# print(f'{evaluate_no(args)}')
print('-----')
print('Clean image multi-label Map : {}'.format(map_clean_ori_multi))
print('Clean image single-label T-Map : {}'.format(map_clean))
print('Aversarial image multi-label Map : {}'.format(map_clean_ori))
print('Aversarial image single-label T-Map(our attack) : {}'.format(map))


