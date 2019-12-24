import torch
import torch.nn as nn
import torchvision
import numpy as np 
import torch_scatter
import util
import cv2
from utils.torch_op import v,npy
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        # m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0,dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,dilation=dilation),
            nn.BatchNorm2d(out_planes,track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,dilation=dilation),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv2d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0,dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,dilation=dilation),
            nn.BatchNorm2d(out_planes,track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,dilation=dilation),
            nn.LeakyReLU(0.1,inplace=True)
        )

class Resnet18_8s(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, args):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = torchvision.models.resnet18(#fully_conv=True,
                                                   pretrained=True,
                                                   #output_stride=32,
                                                   #remove_avg_pool_layer=True
                                                   )
        #resnet18_32s.avgpool = nn.Identity()

        self.args = args
        resnet18_32s.conv1 = nn.Conv2d(args.num_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet_block_expansion_rate = resnet18_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet18_32s.fc = nn.Sequential()
        
        self.resnet18_32s = resnet18_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        #self.segm_layer = nn.Conv2d(32,
        #                           args.snumclass,
        #                           kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        x = self.resnet18_32s.conv1(x)
        x = self.resnet18_32s.bn1(x)
        x = self.resnet18_32s.relu(x)
        x = self.resnet18_32s.maxpool(x)

        x = self.resnet18_32s.layer1(x)
        
        x = self.resnet18_32s.layer2(x)

        logits_8s = self.score_8s(x)
        
        x = self.resnet18_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet18_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]

        logits_16s += nn.functional.interpolate(logits_32s,
                                        size=logits_16s_spatial_dim,mode='bilinear',align_corners=False)
        
        logits_8s += nn.functional.interpolate(logits_16s,
                                        size=logits_8s_spatial_dim,mode='bilinear',align_corners=False)
        
        logits_upsampled = nn.functional.interpolate(logits_8s,
                                        size=input_spatial_dim,mode='bilinear',align_corners=False)

        
        input_spatial_dim_half = [i//2 for i in input_spatial_dim]
        #logits_8s = nn.functional.upsample(logits_8s,size=input_spatial_dim_half,mode='bilinear',align_corners=False)
        #segm=self.segm_layer(logits_upsampled)
        #return logits_upsampled,logits_8s,heatmap
        #return logits_upsampled,segm
        
        if self.args.useTanh:

            logits_upsampled = torch.tanh(logits_upsampled) # scale -1~1

        return logits_upsampled


class SCNet(nn.Module):
    def __init__(self, args):
        super(SCNet, self).__init__()
        ngf=64
        batchnorm=args.batchnorm
        self.useTanh = args.useTanh
        self.skipLayer = args.skipLayer
        self.outputType = args.outputType
        skip_multiplier = 2 if args.skipLayer else 1
        # input is 224x224
        self.conv1rgb = conv2d(batchnorm,4,ngf//2,3,1,1)
        self.conv2rgb = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3rgb = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        self.conv1n = conv2d(batchnorm,4,32,3,1,1)
        self.conv2n = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3n = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        self.conv1d = conv2d(batchnorm,2,ngf//2,3,1,1)
        self.conv2d = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3d = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        inputStream = 3*2

        # now input is 56x56
        self.conv4 = conv2d(batchnorm,ngf*2*inputStream,ngf*4,4,2,1)
        # now input is 28x28
        self.conv5 = conv2d(batchnorm,ngf*4,ngf*8,4,2,1)
        # now input is 14x14
        self.conv6 = conv2d(batchnorm,ngf*8,ngf*8,4,2,1)
        # now input is 7x7
        self.conv7 = conv2d(batchnorm,ngf*8,ngf*8,3,2,0)
        # now input is 3x3
        self.conv8 = conv2d(batchnorm,ngf*8,ngf*8,3,1,1)
        # now input is 3x3
        self.conv9 = conv2d(batchnorm,ngf*8,ngf*16,3,1,0)

        self.deconv9 = deconv2d(batchnorm,ngf*16,ngf*8,3,1,0)
        self.deconv8 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,3,1,1)
        self.deconv7 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,3,2,0)
        self.deconv6 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,4,2,1)
        self.deconv5 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*4,4,2,1)
        self.deconv4 = deconv2d(batchnorm,ngf*4*skip_multiplier,ngf*2,4,2,1)
        
        if 'rgb' in args.outputType:
            self.deconv3rgb=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2rgb=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1rgb=nn.Conv2d(ngf,3,1,1,0)
            self.deconv1rgb.apply(weights_init)
            self.deconv2rgb.apply(weights_init)
            self.deconv3rgb.apply(weights_init)
       
        if 'n' in args.outputType:
            self.deconv3n=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2n=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1n=nn.Conv2d(ngf,3,1,1,0)
            self.deconv1n.apply(weights_init)
            self.deconv2n.apply(weights_init)
            self.deconv3n.apply(weights_init)

        if 'd' in args.outputType:
            self.deconv3d=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2d=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1d=nn.Conv2d(ngf,1,1,1,0)
            self.deconv1d.apply(weights_init)
            self.deconv2d.apply(weights_init)
            self.deconv3d.apply(weights_init)

        if 'k' in args.outputType:
            self.deconv3k=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2k=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1k=nn.Conv2d(ngf,1,1,1,0)
            self.deconv1k.apply(weights_init)
            self.deconv2k.apply(weights_init)
            self.deconv3k.apply(weights_init)

        if 's' in args.outputType:
            self.deconv3s=deconv2d(batchnorm,ngf*2,ngf,4,2,1)
            self.deconv2s=deconv2d(batchnorm,ngf,ngf,4,2,1)
            self.deconv1s=nn.Conv2d(ngf,args.snumclass,1,1,0)
            self.deconv1s.apply(weights_init)
            self.deconv2s.apply(weights_init)
            self.deconv3s.apply(weights_init)

        if 'f' in args.outputType:
            self.deconv3f=deconv2d(batchnorm,ngf*2,ngf,4,2,1)
            self.deconv2f=deconv2d(batchnorm,ngf,ngf,4,2,1)
            self.deconv1f=nn.Conv2d(ngf,32,1,1,0)
            self.deconv1f.apply(weights_init)
            self.deconv2f.apply(weights_init)
            self.deconv3f.apply(weights_init)
        
        self.conv1rgb.apply(weights_init)
        self.conv2rgb.apply(weights_init)
        self.conv3rgb.apply(weights_init)

        self.conv1n.apply(weights_init)
        self.conv2n.apply(weights_init)
        self.conv3n.apply(weights_init)

        self.conv1d.apply(weights_init)
        self.conv2d.apply(weights_init)
        self.conv3d.apply(weights_init)

        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.conv6.apply(weights_init)
        self.conv7.apply(weights_init)
        self.conv8.apply(weights_init)
        self.conv9.apply(weights_init)
        self.deconv7.apply(weights_init)
        self.deconv9.apply(weights_init)
        self.deconv8.apply(weights_init)
        self.deconv7.apply(weights_init)
        self.deconv6.apply(weights_init)
        self.deconv5.apply(weights_init)
        self.deconv4.apply(weights_init)

    def forward(self, x):
        inShape = x.shape[2:]

        x=torch.nn.functional.interpolate(x,[224,224],mode='bilinear',align_corners=False)

        # x:[n,c,h,w]
        # decompose the input into [rgb,normal,depth,mask]
        rgb,norm,depth,mask=x[:,0:3,:,:],x[:,3:6,:,:],x[:,6:7,:,:],x[:,7:8,:,:]
        rgb_t2s,norm_t2s,depth_t2s,mask_t2s=x[:,8+0:8+3,:,:],x[:,8+3:8+6,:,:],x[:,8+6:8+7,:,:],x[:,8+7:8+8,:,:]
        xrgb1 = self.conv1rgb(torch.cat((rgb,mask),1))
        xrgb2 = self.conv2rgb(xrgb1)
        xrgb3 = self.conv3rgb(xrgb2)

        xnorm1 = self.conv1n(torch.cat((norm,mask),1))
        xnorm2 = self.conv2n(xnorm1)
        xnorm3 = self.conv3n(xnorm2)

        xdepth1 = self.conv1d(torch.cat((depth,mask),1))
        xdepth2 = self.conv2d(xdepth1)
        xdepth3 = self.conv3d(xdepth2)

        xrgb1_t2s = self.conv1rgb(torch.cat((rgb_t2s,mask_t2s),1))
        xrgb2_t2s = self.conv2rgb(xrgb1_t2s)
        xrgb3_t2s = self.conv3rgb(xrgb2_t2s)

        xnorm1_t2s = self.conv1n(torch.cat((norm_t2s,mask_t2s),1))
        xnorm2_t2s = self.conv2n(xnorm1_t2s)
        xnorm3_t2s = self.conv3n(xnorm2_t2s)

        xdepth1_t2s = self.conv1d(torch.cat((depth_t2s,mask_t2s),1))
        xdepth2_t2s = self.conv2d(xdepth1_t2s)
        xdepth3_t2s = self.conv3d(xdepth2_t2s)

        
        xin = torch.cat((xrgb3,xrgb3_t2s,xnorm3,xnorm3_t2s,xdepth3,xdepth3_t2s),1)

        x4=self.conv4(xin)
        x5=self.conv5(x4)
        x6=self.conv6(x5)
        x7=self.conv7(x6)
        x8=self.conv8(x7)
        x9=self.conv9(x8)
        
        xout = []

        if self.skipLayer:
            dx9=self.deconv9(x9)
            dx8=self.deconv8(torch.cat((dx9,x8),1))
            dx7=self.deconv7(torch.cat((dx8,x7),1))
            dx6=self.deconv6(torch.cat((dx7,x6),1))
            dx5=self.deconv5(torch.cat((dx6,x5),1))
            dx4=self.deconv4(torch.cat((dx5,x4),1))
            
            if 'rgb' in self.outputType:
                dx3rgb=self.deconv3rgb(torch.cat((dx4,xrgb3),1))
                dx2rgb=self.deconv2rgb(torch.cat((dx3rgb,xrgb2),1))
                dx1rgb=self.deconv1rgb(torch.cat((dx2rgb,xrgb1),1))
                xout.append(dx1rgb)

            if 'n' in self.outputType:
                dx3n=self.deconv3n(torch.cat((dx4,xnorm3),1))
                dx2n=self.deconv2n(torch.cat((dx3n,xnorm2),1))
                dx1n=self.deconv1n(torch.cat((dx2n,xnorm1),1))
                xout.append(dx1n)

            if 'd' in self.outputType:
                dx3d=self.deconv3d(torch.cat((dx4,xdepth3),1))
                dx2d=self.deconv2d(torch.cat((dx3d,xdepth2),1))
                dx1d=self.deconv1d(torch.cat((dx2d,xdepth1),1))
                xout.append(dx1d)

            if 'k' in self.outputType:
                dx3k=self.deconv3k(torch.cat((dx4,xsift3),1))
                dx2k=self.deconv2k(torch.cat((dx3k,xsift2),1))
                dx1k=self.deconv1k(torch.cat((dx2k,xsift1),1))
                xout.append(dx1k)
        else:
            dx9=self.deconv9(x9)
            dx8=self.deconv8(dx9)
            dx7=self.deconv7(dx8)
            dx6=self.deconv6(dx7)
            dx5=self.deconv5(dx6)
            dx4=self.deconv4(dx5)
            
            if 'rgb' in self.outputType:
                dx3rgb=self.deconv3rgb(dx4)
                dx2rgb=self.deconv2rgb(dx3rgb)
                dx1rgb=self.deconv1rgb(dx2rgb)
                xout.append(dx1rgb)

            if 'n' in self.outputType:
                dx3n=self.deconv3n(dx4)
                dx2n=self.deconv2n(dx3n)
                dx1n=self.deconv1n(dx2n)
                xout.append(dx1n)

            if 'd' in self.outputType:
                dx3d=self.deconv3d(dx4)
                dx2d=self.deconv2d(dx3d)
                dx1d=self.deconv1d(dx2d)
                xout.append(dx1d)

            if 'k' in self.outputType:
                dx3k=self.deconv3k(dx4)
                dx2k=self.deconv2k(dx3k)
                dx1k=self.deconv1k(dx2k)
                xout.append(dx1k)

        if 's' in self.outputType:
            dx3s=self.deconv3s(dx4)
            dx2s=self.deconv2s(dx3s)
            dx1s=self.deconv1s(dx2s)
            xout.append(dx1s)

        if 'f' in self.outputType:
            dx3f=self.deconv3f(dx4)
            dx2f=self.deconv2f(dx3f)
            dx1f=self.deconv1f(dx2f)
            if self.useTanh:

                dx1f=torch.tanh(dx1f)
            xout.append(dx1f)

        xout = torch.cat(xout,1)
        xout=torch.nn.functional.interpolate(xout,inShape,mode='bilinear',align_corners=False)

        return xout


class PointNet_TopdownNet(nn.Module):
  def __init__(self, input_chal, output_chal, resolution, nclass):
    super(PointNet_TopdownNet, self).__init__()
    act=None
    act=nn.LeakyReLU
    self.conv1 = conv1d(input_chal, 128, 1, act)
    self.conv2 = conv1d(128, 256, 1, act)
    self.conv3 = conv1d(512, 512, 1, act)
    self.conv4 = conv1d(512, 1024, 1)
    self.conv5 = conv1d(1024, 256, 1)
    self.conv6 = conv1d(256, 64, 1, act=None)
    self.conv7 = conv1d(64, 32, 1, act=None)

    # self.conv_semantic = conv1d(8, 16, 1, act=None)
    self.conv_semantic_pnet = conv1d(32+32, output_chal, 1, act=None)
    self.conv_semantic = nn.Conv2d(32, output_chal, kernel_size=1)
    self.conv_plane = conv1d(64, 4, 1, act=None)
    # self.topdownnet = TopdownNet(input_chal=8*4,ngf=32*2)
    self.topdownnet = TopdownNet(input_chal=64*4,ngf=32,output_chal=output_chal)
    self.drop_layer = nn.Dropout(p=0.5)
    self.resolution = resolution
    self.nclass = nclass
    class tmp():
        pass 
    myargs = tmp()
    myargs.num_input = 3
    myargs.useTanh = False
    self.imgBackbone = Resnet18_8s(myargs)
    #self.imgBackbone_topdown = Resnet18_8s(myargs)
  #def forward(self,roompc, topdown_vis,rgb,imgPCid,img2ind,
  def forward(self,topdown_gt, rgb,imgPCid,img2ind,
  partial,pc2ind,use_predicted_plane):
    
    imgFeat = self.imgBackbone(rgb)
    
    if 1:
        # topdownfeat_gt = self.imgBackbone_topdown(topdown_gt)
        topdownfeat_gt  = None
    else:
        topdownfeat_gt=[]
        for i in range(topdown_gt.shape[0]):
            
            gray= cv2.cvtColor((npy(topdown_gt[i]).transpose(1,2,0)*255).astype('uint8') ,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            step_size = 5
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                                for x in range(0, gray.shape[1], step_size)]
            dense_feat = sift.compute(gray, kp)[1]
            dense_feat = dense_feat.reshape(45,45,128)
            dense_feat= cv2.resize(dense_feat, (224,224))
            topdownfeat_gt.append(dense_feat.transpose(2,0,1))
        
        topdownfeat_gt = np.stack(topdownfeat_gt)
        topdownfeat_gt = v(topdownfeat_gt)

    bindex = v(np.tile(np.arange(rgb.shape[0])[:,None],[1,imgPCid.shape[1]]).reshape(-1)).long()
    features = imgFeat[bindex, :, imgPCid[:,:,1].view(-1).long(), imgPCid[:,:,0].view(-1).long()]
    #features = rgb[bindex, :, imgPCid[:,:,1].view(-1).long(), imgPCid[:,:,0].view(-1).long()]
    pointwise_features = features.view(rgb.shape[0], imgPCid.shape[1], -1).permute(0, 2, 1)
    
    # pointwise_features = self.drop_layer(pointwise_features)
    pc2ind = img2ind

    n = partial.shape[0]
    features = self.conv1(partial)
    features = self.conv2(features)
    features_global = features.max(-1)[0][...,None].repeat(1,1,partial.shape[-1])
    features = torch.cat([features, features_global], dim=1)

    features = self.conv3(features)
    features = self.conv4(features)
    features = self.conv5(features)
    features = self.conv6(features)
    
    features_plane = features.max(2)[0][:,:,None]
    plane_pred = self.conv_plane(features_plane).squeeze(-1)
    plane_pred = torch.cat((plane_pred[:, :3] / torch.norm(plane_pred[:, :3],dim=1,keepdim=True),
                                plane_pred[:, 3:4] ), -1)
    
    if 1:
        pointwise_features_pnet = self.conv7(features)
        pointwise_features = torch.cat((pointwise_features, pointwise_features_pnet), 1)
        # features = partial[:,6:9,:]
        features_out_pnet = self.conv_semantic_pnet(pointwise_features).squeeze(1)
    
    features_out = self.conv_semantic(imgFeat)
    origins = np.zeros([n, 3])
    axis_xs = np.zeros([n, 3])
    axis_ys = np.zeros([n, 3])
    axis_zs = np.zeros([n, 3])
    height = 224
    width = 224
    
    if use_predicted_plane:

        pc2ind = np.zeros([n, partial.shape[-1], 3])
        for i in range(n):
            origin_0 = npy(-plane_pred[i,:3] * plane_pred[i,3])
            # axis [0,0,-1], []
            axis_base = np.array([0,0,-1])
            axis_y_0 = axis_base - np.dot(axis_base,npy(plane_pred[i,:3])) * npy(plane_pred[i,:3])
            axis_y_0 /= (np.linalg.norm(axis_y_0)+1e-16)
            axis_x_0 = np.cross(axis_y_0, npy(plane_pred[i,:3]))
            axis_x_0 /= (np.linalg.norm(axis_x_0)+1e-16)
            axis_z_0 = npy(plane_pred[i,:3])
            origins[i] = origin_0
            axis_xs[i] = axis_x_0
            axis_ys[i] = axis_y_0
            axis_zs[i] = axis_z_0

            pc0 = npy(partial[i,:3,:]).T
            
            
            colors = np.random.rand(self.nclass,3)
            topdown_c_partial_0,_, topdown_ind_0 = util.topdown_projection(pc0, np.ones([pc0.shape[0]]).astype('uint8'), colors, origin_0, axis_x_0, axis_y_0, axis_z_0, height, width, self.resolution)
            pc2ind[i] = topdown_ind_0

        pc2ind = v(pc2ind)
    
    mask_u = (pc2ind[:,:,0] >=0) & (pc2ind[:,:,0]<width)
    mask_v = (pc2ind[:,:,1] >=0) & (pc2ind[:,:,1]<height)
    pointwise_features = pointwise_features.permute(0,2,1)
    mask = mask_u & mask_v
    featgrids = []
    
    for i in range(n):
        feat = pointwise_features[i][mask[i]]
        # index = (pc2ind[i][mask[i]][:,1]*400 + pc2ind[i][mask[i]][:,0]).long()
        index = (pc2ind[i][mask[i]][:,2]*height*width + pc2ind[i][mask[i]][:,1]*width + pc2ind[i][mask[i]][:,0]).long()
        # featgrid = torch_scatter.scatter_mean(feat, index,dim=0,dim_size=400*400).view(400,400,-1)
        # featgrid = torch_scatter.scatter_mean(feat, index,dim=0,dim_size=400*400*4).view(400,400,-1)
        featgrid = torch_scatter.scatter_mean(feat, index,dim=0,dim_size=height*width*4).view(4,height,width,-1)
        # featgrid = torch_scatter.scatter_max(feat, index,dim=0,dim_size=height*width*4,fill_value=0)[0].view(4,height,width,-1)
        featgrids.append(featgrid)
    featgrids = torch.stack(featgrids,0)
    
    # featgrids = featgrids.permute(0,3,1,2)
    featgrids = featgrids.permute(0, 1, 4, 2, 3).contiguous()
    featgrids = featgrids.view(featgrids.shape[0], -1, featgrids.shape[3], featgrids.shape[4])
    
    # torch_scatter.scatter_mean(pointwise_features, (pc2ind[:,:,1]*400 + pc2ind[:,:,0]).unsqueeze(1).clamp(0,10).long(),dim=-1,dim_size=400*400)
    
    #cv2.imwrite('test.png',npy(1-featgrids[1,3:3*2,:,:]).transpose(1,2,0)*255) 
    #cv2.imwrite('test2.png',(1-topdown_vis[1])*255) 
    #util.write_ply('test.ply',npy(partial[1,:3]).T)
    #util.write_ply('test1.ply',npy(roompc[0]))

    topdown_pred,topdown_feat = self.topdownnet(featgrids)
    return topdownfeat_gt, topdown_pred, topdown_feat,features_out, features_out_pnet, plane_pred, v(origins), v(axis_xs),v(axis_ys),v(axis_zs)

class Discriminator(nn.Module):
    def __init__(self, ngpu, input_chal):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            
            nn.Conv2d(input_chal, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.shape[0],-1)

class PlaneNet(torch.nn.Module):
    def __init__(self, input_chal=3, num_s=21, output_chal=7, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(input_chal, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        #self.sy = torch.nn.Sequential(torch.nn.MaxPool1d(num_points), Flatten())
        self.sy = sym_fn
        self.output_chal = output_chal
        self.tnet1 = None
        self.tnet2 = None
        self.num_s = num_s

        self.t_out_t2 = None
        self.t_out_h1 = None
        list_layers = mlp_layers(1088, [512, 256, 128], b_shared=False, bn_momentum=0.1, dropout=0.0)
        # list_layers.append(torch.nn.Linear(64, 2))
        list_layers_s = mlp_layers(1088, [256, 64], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers_s.append(torch.nn.Linear(64, num_s+3+3+32))
        self.regressor_s = torch.nn.Sequential(*list_layers_s)
        
        
  

    def forward(self, x):
        """ x -> features
            [B, 2, c, N] -> [B, K]
        """
        # x = points.transpose(1, 2) # [B, 3, N]


        x_in = x
        b, _, _, num_points = x.shape
        x_cat = torch.cat((x[:,0], x[:,1]))
         
        x_cat = self.h1(x_cat)
        self.t_out_h1 = x_cat # local features

        x_cat = self.h2(x_cat)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x_cat = flatten(self.sy(x_cat))

        l0 = self.t_out_h1 # [B, 64, N]
        g0 = x_cat # [B, K]
        x_cat = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
        #import pdb; pdb.set_trace() 
        pred = self.regressor_s(x_cat.permute(0, 2, 1).contiguous().view(-1, 1088)).view(2*b, num_points, -1)
        
        pred = pred.permute(0, 2, 1)
        
        pred_s = pred[:b].view(b,-1,num_points)
        pred_t = pred[b:].view(b,-1,num_points)

        pred_semantic_s = pred_s[:, :self.num_s, :]
        pred_semantic_t = pred_t[:, :self.num_s, :]

        pred_center_s = pred_s[:, self.num_s:self.num_s+3, :]
        pred_center_t = pred_t[:, self.num_s:self.num_s+3, :]

        pred_norm_s = pred_s[:, self.num_s+3:self.num_s+6, :]
        pred_norm_t = pred_t[:, self.num_s+3:self.num_s+6, :]

        pred_norm_s = pred_norm_s / (torch.norm(pred_norm_s,dim=1,keepdim=True)+1e-16)
        pred_norm_t = pred_norm_t / (torch.norm(pred_norm_t,dim=1,keepdim=True)+1e-16)

        pred_feat_s = pred_s[:, self.num_s+6:, :]
        pred_feat_t = pred_t[:, self.num_s+6:, :]
        
        pred_semantic = torch.cat((pred_semantic_s[:,None,:,:], pred_semantic_t[:,None,:,:]), 1)
        plane_feat = torch.cat((pred_feat_s[:,None,:,:], pred_feat_t[:,None,:,:]), 1)
        plane_pred_n = torch.cat((pred_norm_s[:,None,:,:], pred_norm_t[:,None,:,:]), 1)
        plane_pred_c = torch.cat((pred_center_s[:,None,:,:], pred_center_t[:,None,:,:]), 1)
        plane_pred_n = plane_pred_n.permute(0,1,3,2)
        plane_pred_c = plane_pred_c.permute(0,1,3,2)

        return pred_semantic, plane_feat, plane_pred_n, plane_pred_c



def point_maxpool(inputs, npts, keepdims=False):
    outputs = [torch.max(f, dim=2, keepdim=keepdims)[0]
        for f in torch.split(inputs, npts.tolist(), dim=2)]
    #outputs = [torch.mean(f, dim=2, keepdim=keepdims)
    #    for f in torch.split(inputs, npts.tolist(), dim=2)]
    return torch.cat(outputs, dim=0)

def point_unpool(inputs, npts):
    
    outputs = [f.repeat(1, npts[i]) for i,f in enumerate(inputs)]
    
    return torch.cat(outputs, dim=1)

def conv1d(in_channel, out_channel, kernel_size, act=None,bn=None):
  layers = []
  layers.append(nn.Conv1d(in_channel, out_channel, kernel_size))
  if act is not None:
    layers.append(act(0.1,inplace=True))
  if bn is not None:
    layers.append(nn.BatchNorm1d(out_channel))
  return nn.Sequential(*layers)


def flatten(x):
    return x.view(x.size(0), -1)

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, no_act_bn_at_last=False):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        if i == len(nch_layers)-1 and no_act_bn_at_last:
            pass
        else:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
            layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    #a, _ = torch.max(x, dim=-1, keepdim=True)
    return a

def symfn_avg(x):
    a = torch.nn.functional.avg_pool1d(x, x.size(-1))
    #a = torch.sum(x, dim=-1, keepdim=True) / x.size(-1)
    return a

class PointNet_features(torch.nn.Module):
    def __init__(self, input_chal=3, output_chal=7, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(input_chal, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        #self.sy = torch.nn.Sequential(torch.nn.MaxPool1d(num_points), Flatten())
        self.sy = sym_fn

        self.tnet1 = None
        self.tnet2 = None

        self.t_out_t2 = None
        self.t_out_h1 = None
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, output_chal))
        self.regressor = torch.nn.Sequential(*list_layers)


    def forward(self, x):
        """ x -> features
            [B, 3, N] -> [B, K]
        """
        # x = points.transpose(1, 2) # [B, 3, N]
        # num_points = x.shape[-1]
        x = self.h1(x)
        self.t_out_h1 = x # local features

        x = self.h2(x)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x = flatten(self.sy(x))
        
        x = self.regressor(x)
        # normalize quaternion part 
        
        q_norm = x[:, :4] / torch.norm(x[:, :4],dim=1,keepdim=True)
        t = x[:, 4:7]
        x = torch.cat((q_norm, t), 1)
        # l0 = self.t_out_h1
        # g0 = x
        # x = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
        
        #if self.ret_global:
        #    pass
        #else:
        #    # local + global
        #    l0 = self.t_out_h1 # [B, 64, N]
        #    g0 = x # [B, K]
        #    x = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)

        return x


class RelationNet(torch.nn.Module):
    def __init__(self, input_chal=3, num_s=21, output_chal=7, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(input_chal, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        #self.sy = torch.nn.Sequential(torch.nn.MaxPool1d(num_points), Flatten())
        self.sy = sym_fn
        self.output_chal = output_chal
        self.tnet1 = None
        self.tnet2 = None

        self.t_out_t2 = None
        self.t_out_h1 = None
        list_layers = mlp_layers((1088+32)*2, [512, 256, 64], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(64, 4))
        list_layers_s = mlp_layers(1088, [256, 64], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers_s.append(torch.nn.Linear(64, num_s))
        self.regressor_s = torch.nn.Sequential(*list_layers_s)
        self.regressor = torch.nn.Sequential(*list_layers)
        class tmp():
            pass 
        myargs = tmp()
        myargs.num_input = 3
        myargs.useTanh = False
        self.imgBackbone = Resnet18_8s(myargs)
        self.conv_semantic = nn.Conv2d(32, num_s, kernel_size=1)

    def forward(self, x, pair, img, imgPCid):
        """ x -> features
            [B, 3, N] -> [B, K]
        """
        
        # x = points.transpose(1, 2) # [B, 3, N]
        x_in = x
        b, _, num_points = x.shape
        
        imgFeat = self.imgBackbone(img)
        pred_semantic_img = self.conv_semantic(imgFeat)
        imgFeat_s = imgFeat[:b]
        imgFeat_t = imgFeat[b:]
        bindex = v(np.tile(np.arange(b)[:,None],[1,imgPCid.shape[2]]).reshape(-1)).long()
        features_s = imgFeat_s[bindex, :, imgPCid[:,0,:,1].contiguous().view(-1).long(), imgPCid[:,0,:,0].contiguous().view(-1).long()]
        features_t = imgFeat_t[bindex, :, imgPCid[:,1,:,1].contiguous().view(-1).long(), imgPCid[:,1,:,0].contiguous().view(-1).long()]
        
        x = self.h1(x)
        self.t_out_h1 = x # local features

        x = self.h2(x)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x = flatten(self.sy(x))

        l0 = self.t_out_h1 # [B, 64, N]
        g0 = x # [B, K]
        x = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
        

        
        pred_semantic = self.regressor_s(x.permute(0, 2, 1).contiguous().view(-1, 1088)).view(b, num_points, -1)
        pred_semantic = pred_semantic.permute(0, 2, 1)
        x1 = x[:, :, :num_points//2]
        x2 = x[:, :, num_points//2:]
        
        bindex = torch.arange(b)[:,None].repeat(1, pair.shape[1]).view(-1).long()
        feat1 = x1[bindex, :, pair[:,:,0].view(-1).long()]
        feat2 = x2[bindex, :, pair[:,:,1].view(-1).long()]
          
        feat = torch.cat((feat1, features_s, feat2, features_t), -1)
        pred = self.regressor(feat)
        pred = pred.view(b, -1, 4)
        return pred, pred_semantic, pred_semantic_img

class objectcloud_encoder(nn.Module):
  def __init__(self, input_chal, output_obj=10):
    super(objectcloud_encoder, self).__init__()
    self.output_obj = output_obj
    act=None
    act=nn.LeakyReLU
    self.conv1 = conv1d(input_chal, 128, 1, act)
    self.conv2 = conv1d(128, 256, 1, act)
    self.conv3 = conv1d(512, 512, 1, act)
    self.conv4 = conv1d(512, 1024, 1)
    self.conv5 = conv1d(1024, 512, 1)
    self.conv6 = conv1d(512, 256, 1, act=None)
    self.conv7 = conv1d(256, output_obj*3, 1, act=None)
    self.conv7_conf = conv1d(256, output_obj, 1, act=None)
  def forward(self, partial, npts):
    npts = npts.cpu().numpy().astype('int')
    x = []
    for i, num in enumerate(npts):
        x.append(partial[i, :, :num])
    x = torch.cat(x, 1)
    assert(len(x.shape) == 2)
    x = x.unsqueeze(0)
    features = self.conv1(x)
    features = self.conv2(features)

    features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts).unsqueeze(0)
    features = torch.cat([features, features_global], dim=1)

    features = self.conv3(features)
    features = self.conv4(features)
    
    features = point_maxpool(features, npts)
    
    features = self.conv5(features.unsqueeze(2))
    features = self.conv6(features)
    object_pos = self.conv7(features).view(-1, self.output_obj, 3)
    conf = nn.functional.relu(self.conv7_conf(features).view(-1, self.output_obj))
    return object_pos, conf
