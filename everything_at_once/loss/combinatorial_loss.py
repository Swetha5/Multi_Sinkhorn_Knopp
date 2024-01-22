import torch.nn as nn
import torch
from everything_at_once.loss.contrastive_losses import NormSoftmaxLoss, MMS_Loss
from everything_at_once.model.utils.utils import sim_matrix
import torch.nn.functional as F
import numpy as np
import sys
from collections import Counter


def replace_inf(inp_tensor):
    """Replaces inf by maximum of tensor"""
    ind_inf = torch.nonzero(torch.isinf(inp_tensor))
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 4:
                inp_tensor[ind[0], ind[1], ind[2], ind[3]] = 0
            elif len(ind) == 3:
                inp_tensor[ind[0], ind[1], ind[2]] = 0
            elif len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 4:
                inp_tensor[ind[0], ind[1], ind[2], ind[3]] = m
            elif len(ind) == 3:
                inp_tensor[ind[0], ind[1], ind[2]] = m
            elif len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def check_nan_inf_tensor(input_tensor, var_name):
    check_flag = False
    if torch.isnan(input_tensor).any():
        check_flag = True
    if torch.isinf(input_tensor).any():
        check_flag = True
    return check_flag


def calculate_cost(similarity, expected_codes):
    similarity = similarity.div(0.1).exp()
    return F.binary_cross_entropy_with_logits(similarity, expected_codes)


def multi_sinkhorn(scores, eps=0.002, niters=3, required_properties=32, class_dist=None, discrete=True, fn_name=''):
    """
    Multi-SK algorithm
    """
    B, K = scores.shape
    if discrete:
        MMF = scores.new(np.ones(K))
        MMF[:K-required_properties] = 0.25
    else:
        MMF = ((scores.new(np.arange(0, K)) + 1) / K) * 0.9 + 0.1
    MMFR = MMF.unsqueeze(1).unsqueeze(1).repeat(1, B, K)

    Q = scores.unsqueeze(0).repeat(K, 1, 1) * MMFR
    Q = torch.exp(Q / eps)
    Q = replace_inf(Q)
    Q /= torch.sum(Q)

    M, B, K = Q.shape

    A = Q
    if class_dist is None:
        class_dist = scores.new(np.ones(K)) * 1.0
    else:
        class_dist = scores.new(class_dist)

    class_dist = class_dist / torch.sum(class_dist)
    class_dist = class_dist * B
    class_dist = torch.clamp(class_dist, min=1)
    class_dist = class_dist / torch.sum(class_dist)
    class_dist = class_dist * B

    # sorting
    marginals_argsort = torch.argsort(torch.sum(A[0, :, :], dim=0)).detach()
    class_dist[marginals_argsort] = torch.sort(class_dist)[0]
    class_dist = class_dist.unsqueeze(0).unsqueeze(0).repeat(K, 1, 1)

    min_val_eps = 0.00001
    for _ in range(niters):
        U_SUM = torch.sum(A, dim=1, keepdim=True)
        U_FLAT = class_dist / (U_SUM + min_val_eps)
        U_NORM = U_FLAT.repeat(1, B, 1)
        A = replace_inf(A * U_NORM)

        V_SUM = torch.sum(A, dim=0, keepdim=True)
        V_FLAT = (1.0) / (V_SUM + min_val_eps)
        V_NORM = V_FLAT.repeat(M, 1, 1)
        A = replace_inf(A * V_NORM)

        Z_SUM = torch.sum(A, dim=2, keepdim=True)
        Z_FLAT = (1.0) / (Z_SUM + min_val_eps)
        Z_NORM = Z_FLAT.repeat(1, 1, K)
        A = replace_inf(A * Z_NORM)

    check_nan_inf_tensor(torch.sum(A, dim=1), fn_name+' A sum dim1')
    A = A / torch.sum(A, dim=2, keepdim=True).repeat(1, 1, K)

    return A


def normalize_matrix(A):
    ma, _ = torch.max(A, dim=1, keepdim=True)
    mi, _ = torch.min(A, dim=1, keepdim=True)
    return (A - mi) / (ma - mi)


def calculate_sk(similarity, params={}, fn_name=''):
    """
    compute final assignment matrix
    """
    B, K = similarity.shape
    required_properties = int(K * params['required_properties'])
    discrete = True
    assigment = multi_sinkhorn(similarity, eps=params['eps'], niters=params['niters'], required_properties=required_properties, discrete=discrete, fn_name=fn_name)
    final_assignment = normalize_matrix(torch.sum(assigment[-required_properties:, :, :], dim=0))
    return final_assignment


class AncSimilarityLoss(nn.Module):
    def __init__(self, queue_len, evm_model, sk_params):
        super().__init__()
        text_embed, video_embed, audio_embed = evm_model.cluster_text.embed_dim, evm_model.cluster_video.embed_dim, evm_model.cluster_audio.embed_dim
        # final emb dimenstion is the projection dim (same across modalities)
        final_embed = evm_model.trans_cluster_text.embed_dim
        self.in_text_queue = nn.Parameter(torch.rand(queue_len, text_embed).to('cuda:0'))
        self.in_video_queue = nn.Parameter(torch.rand(queue_len, video_embed).to('cuda:0'))
        self.in_audio_queue = nn.Parameter(torch.rand(queue_len, audio_embed).to('cuda:0'))
        self.out_text_queue = nn.Parameter(torch.rand(queue_len, final_embed).to('cuda:0'))
        self.out_video_queue = nn.Parameter(torch.rand(queue_len, final_embed).to('cuda:0'))
        self.out_audio_queue = nn.Parameter(torch.rand(queue_len, final_embed).to('cuda:0'))

        self.cluster_text = evm_model.cluster_text
        self.cluster_video = evm_model.cluster_video
        self.cluster_audio = evm_model.cluster_audio
        self.trans_cluster_text = evm_model.trans_cluster_text
        self.trans_cluster_video = evm_model.trans_cluster_video
        self.trans_cluster_audio = evm_model.trans_cluster_audio
        self.params = sk_params
        self.beta_loss = self.params['beta_loss'] if 'beta_loss' in self.params else 1.0
        self.evm = evm_model
        self.req_properties = self.params['required_properties']
        self.queue_size = 0
        self.max_queue_len = queue_len

    def forward(self, input_data):
        B, _ = input_data["text_embed_norm"].shape

        # check for text
        check_nan_inf_tensor(input_data["in_text_embed"], 'in_text_embed')
        check_nan_inf_tensor(input_data["text_embed_norm"], 'text_embed_norm')
        check_nan_inf_tensor(self.in_text_queue, 'self.in_text_queue')
        check_nan_inf_tensor(input_data["in_text_assignment"], 'in_text_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.cluster_text.anc_linear.weight, dim=1, p=2),
                             'self.cluster_text.anc_linear.weight')
        check_nan_inf_tensor(self.out_text_queue, 'self.out_text_queue')
        check_nan_inf_tensor(input_data["trans_text_atext_assignment"], 'trans_text_atext_assignment')
        check_nan_inf_tensor(input_data["trans_text_avideo_assignment"], 'trans_text_avideo_assignment')
        check_nan_inf_tensor(input_data["trans_text_aaudio_assignment"], 'trans_text_aaudio_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.trans_cluster_text.anc_linear.weight, dim=1, p=2),
                             'self.trans_cluster_text.anc_linear.weight')
        #check for video
        check_nan_inf_tensor(input_data["in_video_embed"], 'in_video_embed')
        check_nan_inf_tensor(input_data["video_embed_norm"], 'video_embed_norm')
        check_nan_inf_tensor(self.in_video_queue, 'self.in_video_queue')
        check_nan_inf_tensor(input_data["in_video_assignment"], 'in_video_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.cluster_video.anc_linear.weight, dim=1, p=2),
                             'self.cluster_video.anc_linear.weight')
        check_nan_inf_tensor(self.out_video_queue, 'self.out_video_queue')
        check_nan_inf_tensor(input_data["trans_video_atext_assignment"], 'trans_video_atext_assignment')
        check_nan_inf_tensor(input_data["trans_video_avideo_assignment"], 'trans_video_avideo_assignment')
        check_nan_inf_tensor(input_data["trans_video_aaudio_assignment"], 'trans_video_aaudio_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.trans_cluster_video.anc_linear.weight, dim=1, p=2),
                             'self.trans_cluster_video.anc_linear.weight')

        # check for audio
        check_nan_inf_tensor(input_data["in_audio_embed"], 'in_audio_embed')
        check_nan_inf_tensor(input_data["audio_embed_norm"], 'audio_embed_norm')
        check_nan_inf_tensor(self.in_audio_queue, 'self.in_audio_queue')
        check_nan_inf_tensor(input_data["in_audio_assignment"], 'in_audio_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.cluster_audio.anc_linear.weight, dim=1, p=2),
                             'self.cluster_audio.anc_linear.weight')
        check_nan_inf_tensor(self.out_audio_queue, 'self.out_audio_queue')
        check_nan_inf_tensor(input_data["trans_audio_atext_assignment"], 'trans_audio_atext_assignment')
        check_nan_inf_tensor(input_data["trans_audio_avideo_assignment"], 'trans_audio_avideo_assignment')
        check_nan_inf_tensor(input_data["trans_audio_aaudio_assignment"], 'trans_audio_aaudio_assignment')
        check_nan_inf_tensor(nn.functional.normalize(self.trans_cluster_audio.anc_linear.weight, dim=1, p=2),
                             'self.trans_cluster_audio.anc_linear.weight')

        # Update Queues
        with torch.no_grad():
            in_text_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.in_text_queue,
                nn.functional.normalize(self.cluster_text.anc_linear.weight, dim=1, p=2).t()
            ), input_data["in_text_assignment"].detach())), self.params, fn_name='in_text_assignment_for_sk')

            in_video_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.in_video_queue,
                nn.functional.normalize(self.cluster_video.anc_linear.weight, dim=1, p=2).t()
            ), input_data["in_video_assignment"].detach())), self.params, fn_name='in_video_assignment_for_sk')

            in_audio_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.in_audio_queue,
                nn.functional.normalize(self.cluster_audio.anc_linear.weight, dim=1, p=2).t()
            ), input_data["in_audio_assignment"].detach())), self.params, fn_name='in_audio_assignment_for_sk')

            # Transformed assignments for text
            # if self.evm.fp["c_loss_1_1"] > 0:
            trans_text_atext_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_text_queue,
                nn.functional.normalize(self.trans_cluster_text.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_text_atext_assignment"].detach())), self.params, fn_name='trans_text_atext_assignment_for_sk')

            # if self.evm.fp["c_loss_2_1"] > 0:
            trans_text_avideo_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_text_queue,
                nn.functional.normalize(self.trans_cluster_video.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_text_avideo_assignment"].detach())), self.params,
                fn_name='trans_text_avideo_assignment_for_sk')

            trans_text_aaudio_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_text_queue,
                nn.functional.normalize(self.trans_cluster_audio.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_text_aaudio_assignment"].detach())), self.params,
                fn_name='trans_text_aaudio_assignment_for_sk')

            trans_video_atext_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_video_queue,
                nn.functional.normalize(self.trans_cluster_text.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_video_atext_assignment"].detach())), self.params,
                fn_name='trans_video_atext_assignment_for_sk')

            trans_video_avideo_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_video_queue,
                nn.functional.normalize(self.trans_cluster_video.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_video_avideo_assignment"].detach())), self.params,
                fn_name='trans_video_avideo_assignment_for_sk')

            trans_video_aaudio_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_video_queue,
                nn.functional.normalize(self.trans_cluster_audio.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_video_aaudio_assignment"].detach())), self.params,
                fn_name='trans_video_aaudio_assignment_for_sk')

            trans_audio_atext_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_audio_queue,
                nn.functional.normalize(self.trans_cluster_text.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_audio_atext_assignment"].detach())), self.params,
                fn_name='trans_audio_atext_assignment_for_sk')

            trans_audio_avideo_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_audio_queue,
                nn.functional.normalize(self.trans_cluster_video.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_audio_avideo_assignment"].detach())), self.params,
                fn_name='trans_audio_avideo_assignment_for_sk')

            trans_audio_aaudio_assignment_for_sk = calculate_sk(torch.cat((torch.mm(
                self.out_audio_queue,
                nn.functional.normalize(self.trans_cluster_audio.anc_linear.weight, dim=1, p=2).t()
            ), input_data["trans_audio_aaudio_assignment"].detach())), self.params,
                fn_name='trans_audio_aaudio_assignment_for_sk')

            self.in_text_queue[:-B, :] = self.in_text_queue[B:, :].clone()
            self.in_text_queue[-B:, :] = input_data["in_text_embed"].detach()
            self.in_video_queue[:-B, :] = self.in_video_queue[B:, :].clone()
            self.in_video_queue[-B:, :] = input_data["in_video_embed"].detach()
            self.in_audio_queue[:-B, :] = self.in_audio_queue[B:, :].clone()
            self.in_audio_queue[-B:, :] = input_data["in_audio_embed"].detach()

            self.out_text_queue[:-B, :] = self.out_text_queue[B:, :].clone()
            self.out_text_queue[-B:, :] = input_data["text_embed_norm"].detach()
            self.out_video_queue[:-B, :] = self.out_video_queue[B:, :].clone()
            self.out_video_queue[-B:, :] = input_data["video_embed_norm"].detach()
            self.out_audio_queue[:-B, :] = self.out_audio_queue[B:, :].clone()
            self.out_audio_queue[-B:, :] = input_data["audio_embed_norm"].detach()

        loss_dict = {}
        loss = 0
        if self.queue_size < self.max_queue_len:
            self.queue_size += B
        else:
            loss_pair_1_1 = 0
            loss_pair_1_2 = 0
            loss_pair_1_3 = 0
            loss_pair_2_1 = 0
            loss_pair_2_2 = 0
            loss_pair_2_3 = 0
            loss_pair_3_1 = 0
            loss_pair_3_2 = 0
            loss_pair_3_3 = 0
            loss_pair_4_1 = 0
            loss_pair_4_2 = 0
            loss_pair_4_3 = 0
            loss_pair_5_1 = 0
            loss_pair_5_2 = 0
            loss_pair_5_3 = 0
            loss_pair_6_1 = 0
            loss_pair_6_2 = 0
            loss_pair_6_3 = 0
            if self.evm.fp["c_loss_1_1"] > 0:
                loss_pair_1_1 = calculate_cost(input_data["in_text_assignment"], trans_text_atext_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_1_1"]
            if self.evm.fp["c_loss_1_2"] > 0:
                loss_pair_1_2 = calculate_cost(input_data["in_text_assignment"], trans_video_atext_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_1_2"]
            if self.evm.fp["c_loss_1_3"] > 0:
                loss_pair_1_3 = calculate_cost(input_data["in_text_assignment"], trans_audio_atext_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_1_3"]

            loss_dict['loss_pair_1_1'] = loss_pair_1_1
            loss_dict['loss_pair_1_2'] = loss_pair_1_2
            loss_dict['loss_pair_1_3'] = loss_pair_1_3

            loss += loss_pair_1_1
            loss += loss_pair_1_2
            loss += loss_pair_1_3

            if self.evm.fp["c_loss_2_1"] > 0:
                loss_pair_2_1 = calculate_cost(input_data["in_video_assignment"], trans_text_avideo_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_2_1"]
            if self.evm.fp["c_loss_2_2"] > 0:
                loss_pair_2_2 = calculate_cost(input_data["in_video_assignment"], trans_video_avideo_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_2_2"]
            if self.evm.fp["c_loss_2_3"] > 0:
                loss_pair_2_3 = calculate_cost(input_data["in_video_assignment"], trans_audio_avideo_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_2_3"]

            loss_dict['loss_pair_2_1'] = loss_pair_2_1
            loss_dict['loss_pair_2_2'] = loss_pair_2_2
            loss_dict['loss_pair_2_3'] = loss_pair_2_3

            loss += loss_pair_2_1
            loss += loss_pair_2_2
            loss += loss_pair_2_3

            if self.evm.fp["c_loss_3_1"] > 0:
                loss_pair_3_1 = calculate_cost(input_data["in_audio_assignment"], trans_text_aaudio_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_3_1"]
            if self.evm.fp["c_loss_3_2"] > 0:
                loss_pair_3_2 = calculate_cost(input_data["in_audio_assignment"], trans_video_aaudio_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_3_2"]
            if self.evm.fp["c_loss_3_3"] > 0:
                loss_pair_3_3 = calculate_cost(input_data["in_audio_assignment"], trans_audio_aaudio_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_3_3"]

            loss_dict['loss_pair_3_1'] = loss_pair_3_1
            loss_dict['loss_pair_3_2'] = loss_pair_3_2
            loss_dict['loss_pair_3_3'] = loss_pair_3_3
            loss += loss_pair_3_1
            loss += loss_pair_3_2
            loss += loss_pair_3_3

            if self.evm.fp["c_loss_4_1"] > 0:
                loss_pair_4_1 = calculate_cost(input_data["trans_text_atext_assignment"], in_text_assignment_for_sk[-B:,:]) * self.evm.fp["c_loss_4_1"]
            if self.evm.fp["c_loss_4_2"] > 0:
                loss_pair_4_2 = calculate_cost(input_data["trans_text_avideo_assignment"], in_video_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_4_2"]
            if self.evm.fp["c_loss_4_3"] > 0:
                loss_pair_4_3 = calculate_cost(input_data["trans_text_aaudio_assignment"], in_audio_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_4_3"]

            loss_dict['loss_pair_4_1'] = loss_pair_4_1
            loss_dict['loss_pair_4_2'] = loss_pair_4_2
            loss_dict['loss_pair_4_3'] = loss_pair_4_3

            loss += loss_pair_4_1
            loss += loss_pair_4_2
            loss += loss_pair_4_3

            if self.evm.fp["c_loss_5_1"] > 0:
                loss_pair_5_1 = calculate_cost(input_data["trans_video_atext_assignment"], in_text_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_5_1"]
            if self.evm.fp["c_loss_5_2"] > 0:
                loss_pair_5_2 = calculate_cost(input_data["trans_video_avideo_assignment"], in_video_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_5_2"]
            if self.evm.fp["c_loss_5_3"] > 0:
                loss_pair_5_3 = calculate_cost(input_data["trans_video_aaudio_assignment"], in_audio_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_5_3"]

            loss_dict['loss_pair_5_1'] = loss_pair_5_1
            loss_dict['loss_pair_5_2'] = loss_pair_5_2
            loss_dict['loss_pair_5_3'] = loss_pair_5_3

            loss += loss_pair_5_1
            loss += loss_pair_5_2
            loss += loss_pair_5_3

            if self.evm.fp["c_loss_6_1"] > 0:
                loss_pair_6_1 = calculate_cost(input_data["trans_audio_atext_assignment"], in_text_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_6_1"]
            if self.evm.fp["c_loss_6_2"] > 0:
                loss_pair_6_2 = calculate_cost(input_data["trans_audio_avideo_assignment"], in_video_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_6_2"]
            if self.evm.fp["c_loss_6_3"] > 0:
                loss_pair_6_3 = calculate_cost(input_data["trans_audio_aaudio_assignment"], in_audio_assignment_for_sk[-B:, :]) * self.evm.fp["c_loss_6_3"]

            loss_dict['loss_pair_6_1'] = loss_pair_6_1
            loss_dict['loss_pair_6_2'] = loss_pair_6_2
            loss_dict['loss_pair_6_3'] = loss_pair_6_3

            loss += loss_pair_6_1
            loss += loss_pair_6_2
            loss += loss_pair_6_3

            loss = self.beta_loss * loss

        return loss, loss_dict


class CombinatorialLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05,
                 tv_weight=0, ta_weight=0, va_weight=0,
                 t_va_weight=0, v_ta_weight=0, a_tv_weight=0, alpha_closs=1.0, tv_only=False, contrastive_loss_mode='final'):
        super().__init__()

        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()

        self.tv_only = tv_only
        self.tv_weight = tv_weight
        self.ta_weight = ta_weight
        self.va_weight = va_weight
        self.t_va_weight = t_va_weight
        self.v_ta_weight = v_ta_weight
        self.a_tv_weight = a_tv_weight
        self.alpha_closs = alpha_closs
        print(f'alpha_closs {self.alpha_closs}')
        self.contrastive_loss_mode = contrastive_loss_mode
        self.anc_losses = []
        self.losses_names = []

    def set_anc_loss(self, name, anc_loss):
        self.anc_losses.append(anc_loss)
        self.losses_names.append(name)

    def forward(self, input_data):
        loss_info = {}
        nonempty = {}
        nonempty['tv'] = input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']
        if not self.tv_only:
            nonempty['ta'] = input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
            nonempty['va'] = input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']

            nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (
                        input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
            nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (
                        input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
            nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (
                        input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])

        loss_sum = 0
        weight_sum = 0

        contrast_pair_list = [
            ('tv', 'text_embed', 'video_embed', self.tv_weight),
            ('ta', 'text_embed', 'audio_embed', self.ta_weight),
            ('va', 'video_embed', 'audio_embed', self.va_weight),
            ('t_va', 'text_embed', 'va_embed', self.t_va_weight),
            ('v_ta', 'video_embed', 'ta_embed', self.v_ta_weight),
            ('a_tv', 'audio_embed', 'tv_embed', self.a_tv_weight),
        ]

        for name, embed_name1, embed_name2, weight in contrast_pair_list:
            if (embed_name1 in input_data) and (embed_name2 in input_data) and (weight != 0):
                nonempty_mask = nonempty[name]
                embed1 = input_data[embed_name1][nonempty_mask]
                embed2 = input_data[embed_name2][nonempty_mask]

                loss = self.contrastive_loss(sim_matrix(embed1, embed2))
                loss_info[name] = loss.item()
                loss_sum += weight * loss
                weight_sum += weight

        final_loss = self.alpha_closs * (loss_sum / weight_sum)
        loss_info['Retrieval'] = final_loss.item()
        for ind, anc_loss in enumerate(self.anc_losses):
            name = self.losses_names[ind]
            # compute anchor loss
            anc_loss, anc_loss_dict = anc_loss(input_data)
            final_loss += anc_loss
            loss_info[name] = anc_loss
            loss_info.update(anc_loss_dict)
        return final_loss, loss_info