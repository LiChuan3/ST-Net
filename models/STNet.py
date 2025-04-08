import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_with_pos
from layers.StandardNorm import Normalize
from layers.Conv import MultiCycleDecompConv
import math

def FFT_for_seasonal_Period(x, k=3):
    """
    Used for detecting the main cycle of the input sequence in the ITC module.
    """
    with torch.no_grad():
        xf = torch.fft.rfft(x, dim=1)
        frequency_amplitude = abs(xf).mean(dim=0).mean(dim=-1)
        frequency_amplitude[0] = 0
        _, top_indices = torch.topk(frequency_amplitude, k)
        periods = x.shape[1] // top_indices.cpu().numpy()

    xf = torch.fft.rfft(x, dim=1)
    top_amplitudes = abs(xf).mean(-1)[:, top_indices]  # [B, k]
    del xf 
    return periods, top_amplitudes

class DFT_series_decomp(nn.Module):
    """
    The series decomposition method used in ST-block.
    """

    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # [B,T,C]
        x = x.permute(0, 2, 1)  #[B,C,T] 
        xf = torch.fft.rfft(x, dim=2)  # [B,C,T//2+1]

        freq = torch.abs(xf)
        freq[:, :, 0] = 0  # Remove DC 
        topk_freq, _ = torch.topk(freq, self.top_k, dim=2)
        threshold = topk_freq[:, :, -1].unsqueeze(2)
        xf[freq < threshold] = 0

        x_season = torch.fft.irfft(xf, n=x.size(2), dim=2)  # [B,C,T]
        x_trend = x - x_season
        #[B,T,C]
        x_season = x_season.permute(0, 2, 1)
        x_trend = x_trend.permute(0, 2, 1)
        return x_season, x_trend


class ITCs(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.season_top_k = configs.season_top_k
        self.num_layers = configs.num_layers_intra_season

        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiCycleDecompConv(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
                nn.GELU(),
                MultiCycleDecompConv(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
            ) for _ in range(self.num_layers)
        ])
    def single_layer_forward(self, x, layer):
        B, T, N = x.shape
        # FFT
        periods, period_weights = FFT_for_seasonal_Period(x, self.season_top_k)  # [k], [B, k]
        period_res = []
        for i in range(self.season_top_k):
            p = periods[i]
            # 1D -> 2D
            if T % p != 0:
                pad_length = p - (T % p)
                x_pad = F.pad(x, (0, 0, 0, pad_length))
            else:
                pad_length = 0
                x_pad = x

            # Reshape
            num_blocks = (T + pad_length) // p
            x_2d = x_pad.reshape(B, num_blocks, p, N).permute(0, 3, 1, 2)  # [B, N, num_blocks, p]
            # conv
            x_2d = layer(x_2d)
            # 2D -> 1D
            x_1d = x_2d.permute(0, 2, 3, 1).reshape(B, -1, N)
            x_1d = x_1d[:, :T, :]  # trunck

            period_res.append(x_1d)

        # fusion
        period_weights = F.softmax(period_weights, dim=1)
        weighted_res = torch.stack(
            [res * weight.unsqueeze(1).unsqueeze(1)
             for res, weight in zip(period_res, period_weights.unbind(dim=1))],
            dim=-1
        ).sum(dim=-1)

        return weighted_res

    def forward(self, x):
        residual = x 
        for layer in self.layers:
            x = self.single_layer_forward(x, layer) + x
        return x + residual 

class TrendPatchExpert(nn.Module):
    def __init__(self, d_model,  d_ff, patch_size, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Intra-Patch MLP
        self.intra_mlp = nn.Sequential(
            nn.Linear(patch_size * d_model, d_ff),
            #nn.GELU(), w/o Act.
            nn.Dropout(dropout),
            nn.Linear(d_ff, patch_size * d_model),
        )
        # Inter-Patch MLP
        self.inter_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            #nn.GELU(), w/o Act.
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        # Dynamic Weight
        self.weight_gen = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, T, D = x.shape

        # padding
        pad_size = (self.patch_size - (T % self.patch_size)) % self.patch_size
        x_padded = F.pad(x, (0, 0, 0, pad_size))  # [B,T_padded,D]
        T_padded = T + pad_size
        N = T_padded // self.patch_size  # num of Patch

        x_patches = x_padded.view(B, N, self.patch_size, D)  # [B, N, W, D]

        intra_input = x_patches.reshape(B * N, -1)  # [B*N, W*D]
        intra_output = self.intra_mlp(intra_input).view(B, N, self.patch_size, D)

        inter_input = x_patches.mean(dim=2)  # [B, N, D]
        inter_output = self.inter_mlp(inter_input).unsqueeze(2)  # [B, N, 1, D]
        inter_output = inter_output.expand(-1, -1, self.patch_size, -1)  # [B, N, W, D]

        context = x.mean(dim=1)  # [B, D] 
        weight = self.weight_gen(context).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        fused = weight * intra_output + (1 - weight) * inter_output

        output = fused.view(B, T_padded, D)[:, :T, :]  # [B, T, D]

        return output + x

class Inter_Patch_MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, patch_sizes, k=2, dropout=0.3, gate_scales=None):
        super().__init__()
        self.experts = nn.ModuleList([
            TrendPatchExpert(d_model, d_ff, ps, dropout=dropout)
            for ps in patch_sizes
        ])
        self.gate_scales = gate_scales if gate_scales else patch_sizes
        self.num_stats = 3
        gate_input_dim = d_model * (1 + len(self.gate_scales) * self.num_stats)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_experts)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.k = k

    def _extract_scale_features(self, x):
        B, T, D = x.shape
        features = [x.mean(dim=1)]  # GAP

        for ps in self.gate_scales:
            # padding
            pad_size = (ps - (T % ps)) % ps
            x_pad = F.pad(x, (0, 0, 0, pad_size))

            # patch-wise gating information
            patches = x_pad.view(B, -1, ps, D)  # [B, num_patch, ps, D]
            patch_min = patches.amin(dim=2)
            patch_std = patches.std(dim=2)
            patch_max = patches.amax(dim=2)

            # cat
            features.extend([
                patch_min.amin(dim=1),  
                patch_std.mean(dim=1),
                patch_max.amax(dim=1)
            ])

        return torch.cat(features, dim=1)  # [B, gate_input_dim]

    def noisy_topk_gating(self, x):
        gate_input = self._extract_scale_features(x)
        gate_logits = self.gate(gate_input)
        if self.training:
            gate_logits += torch.randn_like(gate_logits) * 0.1
        topk_logits, topk_indices = gate_logits.topk(self.k, dim=-1)  # [B, k]
        topk_gates = self.softmax(topk_logits)  # [B, k]
        return topk_indices, topk_gates

    def forward(self, x):
        B, T, D = x.shape
        topk_indices, topk_gates = self.noisy_topk_gating(x)  # [B, k]
        output = torch.zeros(B, T, D, device=x.device)
        
        for i in range(self.k):
            expert_idx = topk_indices[:, i]  
            gate = topk_gates[:, i]  
            
            unique_experts = expert_idx.unique()
            for expert_id in unique_experts:
                batch_idx = torch.where(expert_idx == expert_id)[0]
                x_subset = x[batch_idx]  
                expert_out = self.experts[expert_id](x_subset)  
                output[batch_idx] += expert_out * gate[batch_idx].view(-1, 1, 1)
        return output

class IPMs(nn.Module):
    def __init__(self, configs):
        super().__init__()
        num_layers = configs.num_layers_intra_trend
        d_model = configs.d_model

        self.layers = nn.ModuleList([
            Inter_Patch_MoE(
                d_model=d_model,
                d_ff=configs.d_ff,
                num_experts=configs.num_experts,
                patch_sizes=configs.patch_sizes,
                k=configs.choose_k,
                dropout=configs.dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x) + x
        return self.norm(x + residual)

class MultiScaleMixer(nn.Module):

    def __init__(self, configs):
        super(MultiScaleMixer, self).__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, series_list):
        mixed_series = []
        current_scale = series_list[0]
        mixed_series.append(current_scale)
        for i, layer in enumerate(self.down_sampling_layers):
            downsampled = layer(current_scale.permute(0, 2, 1)).permute(0, 2, 1)
            if i + 1 < len(series_list):
                next_low = series_list[i + 1] + downsampled
                mixed_series.append(next_low)
                current_scale = next_low
            else:
                mixed_series.append(downsampled)
        return mixed_series

class STblock(nn.Module):
    def __init__(self, configs):
        super(STblock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self.seasonal_feature_extraction = ITCs(configs)

        self.trend_feature_extraction = IPMs(configs)

        self.mixing_multi_scale_series = MultiScaleMixer(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):  # x in x_list , x: [B,T,d_model]
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)
            del season, trend

        out_season_list = [self.seasonal_feature_extraction(s) for s in season_list]  # [B,T,d_model]
        out_trend_list = [self.trend_feature_extraction(t) for t in trend_list]  # [B,T,d_model]

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            out = self.out_cross_layer(out)
            out_list.append(out)
        out_list = self.mixing_multi_scale_series(out_list)
        out_list = [out + x for out, x in zip(out_list, x_list)]
        return out_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.ST_blocks = nn.ModuleList([STblock(configs)
                                         for _ in range(configs.e_layers)])

        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_with_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_with_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.fusion_layer = nn.Linear((configs.down_sampling_layers + 1) * configs.d_model,
                                          configs.d_model)

        if self.channel_independence:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)    # [B,T,d_model]
        out_res = out_res.permute(0, 2, 1)          # [B,d_model,T]
        out_res = self.out_res_layers[i](out_res)   # [B,d_model,T]
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)  # [B,d_model,T]->[B,T,d_model]
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)  # res, moving_mean

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='zeros',
                                  bias=False).to(self.device)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc  # B,C,T
        x_mark_enc_mark_ori = x_mark_enc # B,T,C

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))  # B,T,C
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))  # B,T,C
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])  # :: time step
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list  # B,T,C list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def future_multi_mixing(self, B, enc_out_list):
        dec_out_list = []

        # single prediction result
        for i, enc_out in enumerate(enc_out_list):
            # predict to pred_len
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B*C, pred_len, d_model]
            dec_out_list.append(dec_out)

        # Concat
        concat_dec_out = torch.cat(dec_out_list, dim=-1)  # [B*C, pred_len, (layers+1)*d_model]
        fused_dec_out = self.fusion_layer(concat_dec_out)  # [B*C, pred_len, d_model]

        if self.channel_independence:
            dec_out = self.projection_layer(fused_dec_out)  # [B*C, pred_len, 1]
            # [B, pred_len, C]
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        else:
            dec_out = self.projection_layer(fused_dec_out)  # [B, pred_len, C]

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous()  # [B, N, T]
                    x = x.view(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list)), x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,d_model]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,d_model]
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.ST_blocks[i](enc_out_list)

        # [B * C, length, d_model] -> [B*C,length,1] -> [B,length,C]
        dec_out = self.future_multi_mixing(B, enc_out_list)

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
