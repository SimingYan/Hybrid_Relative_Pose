import os
import config
from utils import train_op

class opts():
  def __init__(self):
    self.parser = train_op.initialize_parser()
    self.parser.add_argument('--arch', default='resnet18',help='specify the network architecture')
    self.parser.add_argument('--pretrain',type=int,default=1,help='')
    self.parser.add_argument('--debug', action='store_true', help = 'debug mode')
    self.parser.add_argument('--batch_size', type = int, default = 8, help = '')
    self.parser.add_argument('--max_epoch', type = int, default=100, help = '')
    self.parser.add_argument('--repeat', type = int, help = '')
    self.parser.add_argument('--batchnorm', type = int, default = 1, help = '')
    self.parser.add_argument('--add_gan_loss', action='store_true', help = '')
    self.parser.add_argument('--ganloss', type = int, default = 0, help = '')
    self.parser.add_argument('--lambda_gan_loss', type=float, default=1e-2, help = '')
    self.parser.add_argument('--pnloss', type = int, default = 0, help = '')
    self.parser.add_argument('--single_view', type = int, default = 1, help = '# ouput')
    self.parser.add_argument('--model',type=str, help='resume ckpt')
    self.parser.add_argument('--featurelearning', type = int, default = 0, help = '')
    self.parser.add_argument('--maskMethod', type = str, default = 'second', help = '')
    self.parser.add_argument('--ObserveRatio',default=0.5,type=float,help='')
    self.parser.add_argument('--outputType', type = str, default = 'rgbdnsf', help = '')
    self.parser.add_argument('--GeometricWeight', type = int, default = 0, help = '')
    self.parser.add_argument('--objectFreqLoss', type = int, default = 0, help = '')
    self.parser.add_argument('--cbw', type = int, default = 0, help = 'class balanced weighting')
    self.parser.add_argument('--dataList', type = str, default = 'matterport3dv1', help = 'options: suncgv3,scannetv1,matterport3dv1')
    self.parser.add_argument('--representation', type = str, default = 'skybox', help = 'options: skybox')
    self.parser.add_argument('--skipLayer', type = int, default = 1, help = '')
    self.parser.add_argument('--snumclass', type = int, default = 21, help = '')
    self.parser.add_argument('--parallel', type = int, default = 0, help = '')
    self.parser.add_argument('--featureDim', type = int, default = 32, help = '')
    self.parser.add_argument('--dynamicWeighting', type = int, default = 0, help = '')
    self.parser.add_argument('--recurrent', type = int, default = 0, help = '')
    self.parser.add_argument('--resize224', type = int, default = 0, help = '') # 1
    self.parser.add_argument('--featlearnSegm', type = int, default = 0, help = '') # 1
    self.parser.add_argument('--useTanh', type = int, default = 1, help = '') # 1
    self.parser.add_argument('--D', type = float, default = 0.5, help = '') # 1
    
    # Siming add
    self.parser.add_argument('--global_exp', type=str, default=None, help='')    
    self.parser.add_argument('--global_pretrain', type=str, default=None, help='')
    self.parser.add_argument('--lr', type=float, default=0.0002, help='')
    self.parser.add_argument('--nViews', type=int, default=2, help='')
    self.parser.add_argument('--reproj', type=int, default=1, help='')
    self.parser.add_argument('--save_pc', type=int, default=0, help='')
    self.parser.add_argument('--planenum', type=int, default=6, help='')
    self.parser.add_argument('--plane_loss', type=str, default='chamfer', help='')
    self.parser.add_argument('--plane_pred_loc', type=int, default=1, help='')
    self.parser.add_argument('--plane_r', type=int, default=1, help='whether using plane representation')
    self.parser.add_argument('--plane_m', type=int, default=0, help='whether using manually generating plane')
    self.parser.add_argument('--pred_plane_size', type=int, default=0, help='whether predict plane size') 
    self.parser.add_argument('--pred_plane_semantic', type=int, default=1, help='whether predict semantic label in plane prediction') 
    self.parser.add_argument('--save_plane', type=int, default=0, help='')
    self.parser.add_argument('--visual_plane', type=int, default=0, help='visualize plane prediction')
    self.parser.add_argument('--topdown_idx', type=int, default=None, help='parallel training for topdown')
    self.parser.add_argument('--plane_idx', type=int, default=None, help='parallel training for old plane prediction')
    self.parser.add_argument('--centerlearning', type=int, default=1, help='whether adding center and normal loss in planenetV2')
    self.parser.add_argument('--normallearning', type=int, default=0, help='whether adding normal loss in planenetV3')
    self.parser.add_argument('--relative_center', type=int, default=0, help='whether predict relative center in planenetV3')
    self.parser.add_argument('--smalldata_debug', type=int, default=0, help='whether using small dataset for debugging, extend the datalist 100 times')

    self.parser.add_argument('--num_workers', type=int, default=8, help='') 
    self.parser.add_argument('--filter_overlap', type=float, default=None, help='')
    self.parser.add_argument('--enable_training', type=int, default=1, help='') 
    self.parser.add_argument('--save_topdown_prediction', type=int, default=0, help='') 

    self.parser.add_argument('--local_method', type=str, default='point', help='') 
    self.parser.add_argument('--eval_local', type=int, default=0, help='') 
    self.parser.add_argument('--local_eval_list', type=str, default='', help='') 
    self.parser.add_argument('--thre_coplane', type=float, default=0.8, help='') 
    self.parser.add_argument('--thre_parallel', type=float, default=0.8, help='') 
    self.parser.add_argument('--thre_perp', type=float, default=0.8, help='') 
    
    
    
  def parse(self):
    self.args = self.parser.parse_args()
    if self.args.debug:
        self.args.num_workers = 0
    return self.args

