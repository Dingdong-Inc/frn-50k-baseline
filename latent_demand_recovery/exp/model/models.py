import os
import torch
from torch import nn
from pypots.imputation import DLinear, TimesNet, PatchTST, SAITS, BRITS, ETSformer, Autoformer, Informer, Transformer, iTransformer, GPVAE, CSDI, ImputeFormer
from pypots.optim import Adam, RAdam
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse, calc_mre, calc_reg_focal, calc_mbe
from pypots.utils.logging import logger
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

def load_model(CONFIG):
    model = CONFIG.get('model', 'TimesNet')
    saving_path = os.path.join(CONFIG['saving_path'], model)
    if model == 'SAITS':
        model = SAITS(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            d_model = CONFIG['d_model'],
            d_ffn = CONFIG['d_ffn'],
            n_heads = CONFIG['n_heads'],
            d_k = CONFIG['d_k'],
            d_v = CONFIG['d_v'],
            dropout = CONFIG['dropout'],
            attn_dropout = CONFIG['attn_dropout'],
            epochs = CONFIG['EPOCHS'],
            batch_size = CONFIG['batch_size'],
            diagonal_attention_mask = True,
            ORT_weight = 1,
            MIT_weight = 1,
            customized_loss_func = calc_mae,
            saving_path = saving_path,
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            patience = CONFIG['patience'],
            OT = CONFIG['OT']
        )
    elif model == 'DLinear':
        model = DLinear(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            moving_avg_window_size = CONFIG['patch_len'] // 2 * 2 + 1,
            individual = False,
            d_model = CONFIG['d_model'],
            ORT_weight = 1,
            MIT_weight = 1,
            batch_size = CONFIG['batch_size'],
            epochs = CONFIG['EPOCHS'],
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            saving_path = saving_path,
            patience = CONFIG['patience'],
            OT = CONFIG['OT']
        )
    elif model == 'TimesNet':
        model = TimesNet(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            top_k = 7,
            d_model = CONFIG['d_model'],
            d_ffn = CONFIG['d_ffn'],
            n_kernels = 5,
            dropout = CONFIG['dropout'],
            apply_nonstationary_norm = True,
            epochs = CONFIG['EPOCHS'],
            batch_size = CONFIG['batch_size'],
            saving_path = saving_path,
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            patience = CONFIG['patience'],
            OT = CONFIG['OT']
        )
    elif model == 'iTransformer':
        model = iTransformer(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            d_model = CONFIG['d_model'],
            n_heads = CONFIG['n_heads'],
            d_k = CONFIG['d_k'],
            d_v = CONFIG['d_v'],
            d_ffn = CONFIG['d_ffn'],
            dropout = CONFIG['dropout'],
            attn_dropout = CONFIG['attn_dropout'],
            ORT_weight = 1,
            MIT_weight = 1,
            batch_size = CONFIG['batch_size'],
            epochs = CONFIG['EPOCHS'],
            patience = CONFIG['patience'],
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            saving_path = saving_path,
            OT = CONFIG['OT']
        )
    elif model == 'GPVAE':
        model = GPVAE(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            latent_size = 4,
            encoder_sizes = (64, 64),
            decoder_sizes = (64, 64),
            kernel = "cauchy",
            beta = 0.01,
            M = 1,
            K = 1,
            sigma = 1.0,
            length_scale = 48.0,
            kernel_scales = 3,
            window_size = 3,
            batch_size = CONFIG['batch_size'],
            epochs = CONFIG['EPOCHS'],
            patience = CONFIG['patience'],
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            saving_path = saving_path,
        )
    elif model == 'CSDI':
        model = CSDI(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            n_heads = CONFIG['n_heads'],
            n_channels = CONFIG['d_ffn'], 
            d_time_embedding = CONFIG['d_model'],
            d_feature_embedding = CONFIG['d_model'],
            d_diffusion_embedding = CONFIG['d_model'],
            n_diffusion_steps = 30,
            target_strategy = "random",
            schedule = 'linear',
            beta_start = 0.0001,
            beta_end = 0.04,
            is_unconditional = False,
            batch_size = CONFIG['batch_size'],
            epochs = CONFIG['EPOCHS'],
            patience = CONFIG['patience'],
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            saving_path = saving_path            
        )
    elif model == 'ImputeFormer':
        model = ImputeFormer(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            d_input_embed = CONFIG['d_model'],
            d_learnable_embed = CONFIG['d_model'],
            d_proj = CONFIG['d_model'],
            d_ffn = CONFIG['d_ffn'],
            n_temporal_heads = CONFIG['n_heads'],
            dropout = CONFIG['dropout'],
            input_dim = 1,
            output_dim = 1,
            ORT_weight = 1,
            MIT_weight = 1,
            batch_size = CONFIG['batch_size'],
            epochs = CONFIG['EPOCHS'],
            patience = CONFIG['patience'],
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            saving_path = saving_path,
        )
    else:
        raise NotImplementedError(f'{model} is not implemented')
    return model