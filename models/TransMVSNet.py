import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .FMT import FMT_with_pathway
import pdb

Align_Corners_Range = False

class PixelwiseNet(nn.Module):

    def __init__(self):

        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1):
        """forward.

        :param x1: [B, 1, D, H, W]
        """

        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1) # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0] # [B, 1, H ,W]

        return output


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self, features, proj_matrix, depth_values, num_depth, cost_regularization, \
        view_weights=None):
        """forward.

        :param stage_i: int, index of stage, [1, 2, 3], stage_1 corresponds to lowest image resolution
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrix: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, regularization network
        :param view_weights: pixel wise view weights for src views
        """

        # (Pdb) len(features) -- 5
        # (Pdb) features[0].size() -- [3, 32, 216, 288]
        # (Pdb) features[1].size() -- [3, 32, 216, 288]
        # (Pdb) features[2].size() -- [3, 32, 216, 288]
        # (Pdb) features[3].size() -- [3, 32, 216, 288]
        # (Pdb) features[4].size() -- [3, 32, 216, 288]
        # proj_matrix.size() -- [3, 5, 2, 4, 4]
        proj_matrix = torch.unbind(proj_matrix, 1) # len(torch.unbind(proj_matrix, 1)) -- 5

        assert len(features) == len(proj_matrix), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrix[0], proj_matrix[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            similarity_sum += similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
            pixel_wise_weight_sum += view_weight.unsqueeze(1) # [B, 1, 1, H, W]

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]

        # step 3. cost volume regularization
        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photo_confidence = torch.max(prob_volume, dim=1)[0]

        if view_weights == None: # True
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "photo_confidence": photo_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth,  "photo_confidence": photo_confidence, "prob_volume": prob_volume, "depth_values": depth_values}


class TransMVSNet(nn.Module):
    def __init__(self, ndepths=[48, 32, 8],
                 depth_interals_ratio=[4.0, 1.0, 0.5],
                 cr_base_chs=[8, 8, 8]):
        super(TransMVSNet, self).__init__()
        # ndepths = [48, 32, 8]
        # depth_interals_ratio = [4.0, 1.0, 0.5]
        # cr_base_chs = [8, 8, 8]

        assert len(ndepths) == len(depth_interals_ratio)

        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)

        self.stage_scales = {
            "stage1": 4.0,
            "stage2": 2.0,
            "stage3": 1.0
        }

        self.feature = FeatureNet(base_channels=8)
        self.FMT_with_pathway = FMT_with_pathway()
        self.cost_regularization = nn.ModuleList([ \
                CostRegNet(in_channels=1, base_channels=self.cr_base_chs[i]) \
                    for i in range(self.num_stage)])
        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrix, depth_values):
        # imgs.size() -- [3, 5, 3, 864, 1152]
        # proj_matrix.keys() -- dict_keys(['stage1', 'stage2', 'stage3'])
        # depth_values.size() -- [3, 192]
        S, B, C, H, W = imgs.size() # S -- Stage

        depth_min = float(depth_values[0, 0].cpu().numpy()) # 425.00
        depth_max = float(depth_values[0, -1].cpu().numpy()) # 931.45
        depth_interval = (depth_max - depth_min) / depth_values.size(1) # 2.6362

        # step 1. feature extraction
        features = []
        # imgs.size() -- [3, 5, 3, 864, 1152]
        for nview_idx in range(imgs.size(1)): # 5
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        # img.size() -- [3, 3, 864, 1152]
        # (Pdb) features[0]['stage1'].size() -- [3, 32, 216, 288]
        # (Pdb) features[0]['stage2'].size() -- [3, 16, 432, 576]
        # (Pdb) features[0]['stage3'].size() -- [3, 8, 864, 1152]

        features = self.FMT_with_pathway(features)

        outputs = {}
        depth, cur_depth = None, None
        view_weights = None

        for stage_i in range(self.num_stage): # 3
            stage_n = "stage{}".format(stage_i + 1)
            state_features = [feat[stage_n] for feat in features]
            state_proj_matrix = proj_matrix[stage_n]
            stage_scale = self.stage_scales[stage_n]

            if depth is not None: # stage 2/3
                cur_depth = depth.detach()
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                        [img.shape[2], img.shape[3]], mode='bilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_samples = get_depth_samples(cur_depth=cur_depth,
                    ndepth=self.ndepths[stage_i],
                    depth_inteval_pixel=self.depth_interals_ratio[stage_i] * depth_interval,
                    dtype=img[0].dtype,
                    device=img[0].device,
                    shape=[img.shape[0], img.shape[2], img.shape[3]],
                    max_depth=depth_max,
                    min_depth=depth_min)
            # depth_samples.size() -- [3, 48, 864, 1152]

            if stage_i + 1 > 1: # for stage 2 and 3
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

            # self.ndepths=[48, 32, 8]
            # stage_scale -- 4.0, 2.0, 1.0
            if view_weights == None: # stage 1
                outputs_stage, view_weights = self.DepthNet(
                        state_features,
                        state_proj_matrix,
                        depth_values=F.interpolate(depth_samples.unsqueeze(1), 
                            [self.ndepths[stage_i], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)],
                             mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_i],
                        cost_regularization=self.cost_regularization[stage_i], view_weights=view_weights)
            else:
                outputs_stage = self.DepthNet(
                        state_features,
                        state_proj_matrix,
                        depth_values=F.interpolate(depth_samples.unsqueeze(1), 
                            [self.ndepths[stage_i], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)],
                             mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_i],
                        cost_regularization=self.cost_regularization[stage_i], view_weights=view_weights)

            wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
            depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)

            # depth hypotheses 425mm to 935mm
            outputs_stage['depth'] = depth.clamp(425.0, 935.0)

            outputs[stage_n] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
