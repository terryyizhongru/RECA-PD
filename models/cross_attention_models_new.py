import torch
from einops.layers.torch import Reduce
from .multiheaded_attention import MultiHeadedAttention
import torch.nn as nn


class RECAPD_Model(torch.nn.Module):
    """SSL Embedding & Temporal Cross-Attention Model.
    """

    def __init__(self, config):
        super(RECAPD_Model, self).__init__()

        self.config = config
        self.attn_type = self.config.model
        
        # -- computing informed-based speech features input dimension
        informed_input_dim = 0
        for feature in self.config.features:
            informed_input_dim += feature['input_dim']

        self.feature_nets = torch.nn.ModuleDict()
        for feat in config.features:
            name = feat['name']
            in_dim = feat['input_dim']
            # linear + activation to map to D
            self.feature_nets[name] = self.make_three_layer_net(in_dim, self.config.ssl_features_conf['input_dim'])
            print(self.feature_nets[name])
        # -- model architecture setup
        # 'cross_embed': [DxT] · [TxF] --> [DxF] --> softmax([FxD]) · [DxT] --> [FxT] --> mean([FxT]) --> [F] --
        #                                                                                                                   --> Classification
        # 'cross_time':  [TxD] · [DxF] --> [TxF] --> softmax([FxT]) · [TxD] --> [FxD] --> mean([FxD]) --> [F] --
        self.query_dim = self.config.ssl_features_conf['input_dim']
        self.key_dim = self.config.ssl_features_conf['input_dim']
        self.value_dim = self.config.ssl_features_conf['input_dim']
        self.D = self.config.ssl_features_conf['input_dim']

        self.cross_attn = MultiHeadedAttention(
            query_dim=self.D,
            key_dim=self.D,
            value_dim=self.D,
            num_heads=config.model_conf['num_heads'],
            dropout_rate=config.model_conf['dropout'],
            attn_type='new',
        )
        # # -- embedding cross attention
        # self.embed_mha = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=self.config.model_conf['num_heads'],
        #     dropout_rate=self.config.model_conf['dropout'],
        #     attn_type='cross_embed',
        # )
        
        # self.cross_attn = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=config.model_conf['num_heads'],
        #     dropout_rate=config.model_conf['dropout'],
        #     attn_type='cross_time',  # new type if necessary
        # )

        # # -- temporal cross attention
        # self.time_mha = MultiHeadedAttention(
        #     query_dim=self.query_dim,
        #     key_dim=self.key_dim,
        #     value_dim=self.value_dim,
        #     num_heads=self.config.model_conf['num_heads'],
        #     dropout_rate=self.config.model_conf['dropout'],
        #     attn_type='cross_time',
        # )

        # -- classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.key_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.key_dim,
                self.config.num_classes,
                bias=False,
            ),
        )

        # 6. Computing Loss Function
        if config.training_settings['loss_criterion'] == 'cross_entropy':
            self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError(f'unknown loss criterion {config.training_settings["loss_criterion"]}')

    def forward(self, batch):
        model_output = {}
        # import pdb; pdb.set_trace()
        B, T, D = batch[self.config.ssl_features].shape
        
        tokens = []
        for feat in self.config.features:
            name = feat['name']
            x = batch[name]           # (B, 1, in_dim)
            x = x.squeeze(1)          # (B, in_dim)
            tok = self.feature_nets[name](x)  # (B, D)
            tokens.append(tok)
            
        tokens = torch.stack(tokens, dim=1)  # (B, K, D)
        
        Q = batch[self.config.ssl_features]  # (B, T, D)
        # cross attention: Q x (K,D) tokens
        out = self.cross_attn(Q, tokens, tokens, mask=batch['mask_ssl'])  # (B, T, D)
        # aggregate over time
        repr = Reduce('b n d -> b d', 'mean')(out)                               # (B, D)                                   # (B, D)
        # classification
        logits = self.classifier(repr).squeeze(1)                        # (B, num_classes)        


        #========================
        # build model_output
        model_output['subject_id'] = batch['subject_id']
        model_output['sample_id']  = batch['sample_id']
        model_output['embeddings'] = repr
        model_output['logits']     = logits
        model_output['probs']      = torch.nn.functional.softmax(logits, dim=-1)
        model_output['preds']      = logits.argmax(dim=-1)
        model_output['labels']     = batch['label']
        model_output['loss']       = self.loss_criterion(logits, batch['label'])
        model_output['attn_scores']= self.cross_attn.attn_scores
        return model_output


    def make_three_layer_net(self, in_dim: int, target_dim: int = 1024,
                            mid1: int = 128, mid2: int = 512,
                            use_norm: bool = True, p_drop: float = 0.1):
        """
        3‑层 MLP: in_dim → mid1 → mid2 → target_dim
        每层 SiLU，选配 LayerNorm + Dropout
        """
        layers = []

        # ① in_dim → mid1
        layers.append(nn.Linear(in_dim, mid1, bias=False))
        if use_norm:
            layers.append(nn.LayerNorm(mid1))
        layers.append(nn.SiLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))

        # ② mid1 → mid2
        layers.append(nn.Linear(mid1, mid2, bias=False))
        if use_norm:
            layers.append(nn.LayerNorm(mid2))
        layers.append(nn.SiLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))

        # ③ mid2 → target_dim
        layers.append(nn.Linear(mid2, target_dim, bias=False))
        layers.append(nn.SiLU())          # 若要保持线性输出，可把这一行删掉

        return nn.Sequential(*layers)
