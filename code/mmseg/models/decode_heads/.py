import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
import random,os,copy
from mmcv.cnn import build_norm_layer
import cv2 as cv
from PIL import Image
class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())



    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class VIT_MLAHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,num_conv=2,upsampling_method='bilinear',
                 norm_layer=nn.BatchNorm2d, norm_cfg=None,loss_edge_decode=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=0.4),
                 use_edge_loss=True,
                 **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.use_edge_loss = use_edge_loss
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_conv = num_conv
        self.upsampling_method = upsampling_method

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels,
                             self.num_classes, 3, padding=1)
        out_channel = self.num_classes
        self.conv_0 = nn.Conv2d(512, 256, 1, 1)
        self.conv_1 = nn.Conv2d(512, out_channel, 1, 1)
        self.conv_2 = nn.Conv2d(256, out_channel, 1, 1)
        self.loss_edge_decode = build_loss(loss_edge_decode)
    def featuremap_2_heatmap(self, feature_map):
        assert isinstance(feature_map, torch.Tensor)
        feature_map = feature_map.detach()
        heatmap = feature_map[:, 0, :, :] * 0
        for c in range(feature_map.shape[1]):
            heatmap += feature_map[:, c, :, :]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # heatmap=np.ones(heatmap.shape)-heatmap
        return heatmap

    def forward(self, inputs):
        y = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        y = self.conv_0(y)
        z=self.conv_2(y)
        z = F.interpolate(
            z, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        # for i in range(2):
        #     c1 = z[:, i:i + 1, :, :]
        #     filename = "/opt/data/private/SETR/code/SETR-main_mla/fig/edge_map/" + str(i + 1) + ".jpg"
        #     heatmap = self.featuremap_2_heatmap(c1)
        #     heatmap = np.uint8(255 * heatmap)
        #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        #     cv.imwrite(filename, heatmap)

        # t=z[0].cpu().numpy()
        # t=t.transpose(1, 2, 0)
        # t=t[:,:,0]
        # t=np.array([t for i in range(3)]).transpose(1, 2, 0)
        # print('t=', t.shape, type(t))
        # t = Image.fromarray(np.uint8(t))
        # t.save('/opt/data/private/SETR/code/SETR-main_mla/fig/edge_labels_image.png')


        # x = self.cls(x)
        # x = F.interpolate(x, size=self.img_size, mode='bilinear',
        #                   align_corners=self.align_corners)
        x=inputs[3]
        x = F.interpolate(
            x, size=x.shape[-1] * 4, mode='bilinear', align_corners=self.align_corners)
        x=torch.cat([x,y], dim=1)
        # for i in range(512):
        #     c1 = x[:, i:i + 1, :, :]
        #     filename = "/opt/data/private/SETR/code/SETR-main_mla/fig/feature_map/" + str(i + 1) + ".jpg"
        #     heatmap = self.featuremap_2_heatmap(c1)
        #     heatmap = np.uint8(255 * heatmap)
        #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        #     cv.imwrite(filename, heatmap)
        x = self.conv_1(x)
        x = F.interpolate(
            x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        # for i in range(2):
        #     c1 = x[:, i:i + 1, :, :]
        #     filename = "/opt/data/private/SETR/code/SETR-main_mla/fig/x_map/" + str(i + 1) + ".jpg"
        #     heatmap = self.featuremap_2_heatmap(c1)
        #     heatmap = np.uint8(255 * heatmap)
        #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        #     cv.imwrite(filename, heatmap)
        if self.use_edge_loss:
            return x, z
        else:
            return x

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, ignore se_loss."""
        if self.use_edge_loss:
            return self.forward(inputs)[0]
        else:
            return self.forward(inputs)

    @staticmethod
    def _convert_to_edge_labels(seg_label):
        # print(seg_label.shape, type(seg_label))
        batch_size = seg_label.size(0)
        edge_labels = seg_label.new_zeros(seg_label.size(0),seg_label.size(2),seg_label.size(3))
        # print(edge_labels.shape,type(edge_labels))
        for i in range(batch_size):
            # img_cv = cv.imread('/opt/data/private/img_dir/natural/CASIA2/tamper/train/gt/Tp_D_CND_S_N_ani00073_ani00068_00193.png')
            # print(img_cv.shape, type(img_cv))
            # ekk = cv.Canny(img_cv, 0, 1)
            # print(ekk.shape, type(ekk))
            # ekk = np.array([ekk for i in range(3)]).transpose(1, 2, 0)
            # print(ekk.shape, type(ekk))
            # ekk = Image.fromarray(ekk)
            # ekk.save('/opt/data/private/SETR/code/SETR-main_mla/mmseg/models/decode_heads/ekk.png')
            # print(ekk.shape, type(ekk))
            z=seg_label[i].permute(1,2,0).cpu().numpy()
            z=z.reshape(z.shape[0],z.shape[1])
            z=np.array([z for i in range(3)]).transpose(1, 2, 0)
            z_original=copy.deepcopy(z)
            z_original[z_original==1] = 0
            z[z >= 255] = 0
            # # # # # # # # # # # ## # # # # # # # # # # ## # # # # # # # # # # #
            # before = Image.fromarray(np.uint8(z_original1))
            # demo_dir = "/opt/data/private/SETR/code/SETR-main_mla/fig/canny/"
            # demo_path_before = os.path.join(demo_dir, "before"+ str(1) + ".png")
            # demo_path_after = os.path.join(demo_dir,  "after"+str(1) + ".png")
            # before.save(demo_path_before)
            edge_labels_image= cv.Canny(np.uint8(z),0,1)/255+ z_original[:,:,1]

            # after = np.array([edge_labels_image for i in range(3)]).transpose(1, 2, 0)
            # after = Image.fromarray(np.uint8(after))
            # after.save(demo_path_after)
            ## # # # # # # # # # # ## # # # # # # # # # # ## # # # # # # # # # # ## # # # # # # # # # # #
            edge_labels[i]=torch.from_numpy(edge_labels_image)
        # print('edge_labels=', edge_labels.shape)
        return edge_labels
    # def resize(input,
    #            size=None,
    #            scale_factor=None,
    #            mode='nearest',
    #            align_corners=None,
    #            warning=True):
    #     if warning:
    #         if size is not None and align_corners:
    #             input_h, input_w = tuple(int(x) for x in input.shape[2:])
    #             output_h, output_w = tuple(int(x) for x in size)
    #             if output_h > input_h or output_w > output_h:
    #                 if ((output_h > 1 and output_w > 1 and input_h > 1
    #                      and input_w > 1) and (output_h - 1) % (input_h - 1)
    #                         and (output_w - 1) % (input_w - 1)):
    #                     warnings.warn(
    #                         f'When align_corners={align_corners}, '
    #                         'the output would more aligned if '
    #                         f'input size {(input_h, input_w)} is `x+1` and '
    #                         f'out size {(output_h, output_w)} is `nx+1`')
    #     if isinstance(size, torch.Size):
    #         size = tuple(int(x) for x in size)
    #     return F.interpolate(input, size, scale_factor, mode, align_corners)
    def losses(self, seg_logit, seg_label):
        """Compute segmentation and semantic encoding loss."""
        seg_logit, se_seg_logit = seg_logit
        # print('seg_logit=', seg_logit.shape, type(seg_logit))
        # print('se_seg_logit=', se_seg_logit.shape, type(se_seg_logit))
        # se_seg_logit = resize(
        #     input=se_seg_logit,
        #     size=se_seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # print(se_seg_logit.shape)
        loss = dict()
        loss.update(super(VIT_MLAHead, self).losses(seg_logit, seg_label))
        edge_loss = self.loss_edge_decode(
            se_seg_logit,
            self._convert_to_edge_labels(seg_label))
        loss['loss_edge'] = edge_loss
        return loss
