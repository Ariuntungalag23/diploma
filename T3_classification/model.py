"""
================================================================================
MODEL — ConvNeXt-Base + EMA + Layer-wise LR Decay optimizer
================================================================================
"""

from copy import deepcopy

import timm
import torch
import torch.nn as nn

from config import CFG


# ============================================================================
# FRACTURE CLASSIFIER MODEL
# ============================================================================

class FractureModel(nn.Module):
    """
    ConvNeXt-Base + LayerNorm + Dropout + Linear head
    
    Binary classification → 1 logit гаргана (BCEWithLogitsLoss-д тохиромжтой)
    """
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=True,
            num_classes=0,              # Тэр чигт нь head устгана
            drop_path_rate=cfg.drop_path_rate,
            global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, 1),     # Binary logit
        )
    
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(-1)  # (B,)
        return logits


# ============================================================================
# EMA (EXPONENTIAL MOVING AVERAGE)
# ============================================================================

class ModelEMA:
    """
    Model weight-ийн EMA (exponential moving average).
    
    Сургалтын явцад weights-ийг тогтоон averaging хийнэ → 
    илүү тогтвортой, ерөнхийдөө +0.3-0.7% AUC өсдөг.
    
    decay = 0.9995 → 1/(1-0.9995) = 2000 алхамын window
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        # Deepcopy → original model-аас тусдаа параметртэй EMA model
        self.module = deepcopy(self._unwrap(model)).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)
    
    @staticmethod
    def _unwrap(model):
        """torch.compile эсвэл DataParallel-аар wrap хийгдсэн бол задлах"""
        if hasattr(model, '_orig_mod'):
            return model._orig_mod
        if hasattr(model, 'module'):
            return model.module
        return model
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Нэг алхам шинэчлэх: ema = decay * ema + (1-decay) * model"""
        d = self.decay
        model_sd = self._unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + model_sd[k].detach() * (1.0 - d))


# ============================================================================
# LAYER-WISE LR DECAY OPTIMIZER
# ============================================================================

def get_layer_id_convnext(name: str, num_stages: int = 4) -> int:
    """
    ConvNeXt parameter name → layer ID
    
    ConvNeXt-Base архитектур:
        backbone.stem           → layer 0 (хамгийн доод, хамгийн бага LR)
        backbone.stages.0.*     → layer 1
        backbone.stages.1.*     → layer 2
        backbone.stages.2.*     → layer 3
        backbone.stages.3.*     → layer 4
        head.*                  → layer 5 (хамгийн дээд, full LR)
    """
    if name.startswith('head'):
        return num_stages + 1
    
    if 'stages' in name:
        # "backbone.stages.2.blocks.5.norm.weight" → stage id = 2 → layer 3
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'stages' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1]) + 1
                except ValueError:
                    pass
    
    # Stem эсвэл бусад → layer 0
    return 0


def build_optimizer(model: nn.Module, cfg: CFG, num_stages: int = 4):
    """
    AdamW + Layer-wise LR decay (LLD)
    
    Доод layer-ууд (stem, stages.0) → бага LR (pretrained feature хадгалах)
    Дээд layer-ууд (stages.3, head) → их LR (fracture-д тохируулах)
    
    LR scaling:
        head     : cfg.lr_head (5e-4)
        stage 4  : cfg.lr_backbone × decay^0 = 5e-5
        stage 3  : cfg.lr_backbone × decay^1 = 3.75e-5
        stage 2  : cfg.lr_backbone × decay^2 = 2.81e-5
        stage 1  : cfg.lr_backbone × decay^3 = 2.11e-5
        stem     : cfg.lr_backbone × decay^4 = 1.58e-5
    """
    decay = cfg.layer_decay
    
    # Bias болон normalization layer-уудад weight decay хэрэглэхгүй
    no_decay_keywords = ('bias', 'norm', 'gamma', 'beta')
    
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        layer_id = get_layer_id_convnext(name, num_stages)
        
        # LR scaling
        if layer_id >= num_stages + 1:
            lr = cfg.lr_head
        else:
            scale = decay ** (num_stages - layer_id)
            lr = cfg.lr_backbone * scale
        
        # Weight decay
        wd = 0.0 if any(nd in name for nd in no_decay_keywords) else cfg.weight_decay
        
        params.append({
            'params': p,
            'lr': lr,
            'weight_decay': wd,
            'layer_id': layer_id,
            'name': name,  # debug-д хэрэгтэй
        })
    
    optimizer = torch.optim.AdamW(params)
    return optimizer
