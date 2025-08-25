import torch
from einops.layers.torch import Reduce
from .multiheaded_attention_fixnorm import MultiHeadedAttention

class CrossFullModelFix(torch.nn.Module):
    """SSL Embedding & Temporal Cross-Attention Model.
    """

    def __init__(self, config):
        super(CrossFullModelFix, self).__init__()

        self.config = config
        self.attn_type = self.config.model

        # -- computing informed-based speech features input dimension
        informed_input_dim = 0
        for feature in self.config.features:
            informed_input_dim += feature['input_dim']

        # -- model architecture setup
        # 'cross_embed': [DxT] · [Tx619] --> [Dx619] --> softmax([619xD]) · [DxT] --> [619xT] --> mean([619xT]) --> [619] --
        #                                                                                                                   --> Classification
        # 'cross_time':  [TxD] · [Dx619] --> [Tx619] --> softmax([619xT]) · [TxD] --> [619xD] --> mean([619xD]) --> [619] --
        self.query_dim = self.config.ssl_features_conf['input_dim']
        self.key_dim = informed_input_dim
        self.value_dim = self.config.ssl_features_conf['input_dim']

        # -- embedding cross attention
        self.embed_mha = MultiHeadedAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_heads=self.config.model_conf['num_heads'],
            dropout_rate=self.config.model_conf['dropout'],
            attn_type='cross_embed',
        )

        # -- temporal cross attention
        self.time_mha = MultiHeadedAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_heads=self.config.model_conf['num_heads'],
            dropout_rate=self.config.model_conf['dropout'],
            attn_type='cross_time',
        )

        # -- classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.key_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.key_dim * 2,
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

        # -- concatenating all informed-based speech features
        all_informed_features = []
        for feature in self.config.features:
            feature_id = feature['name']
            informed_data = batch[feature_id] # -- (batch, 1, input_dim)
            all_informed_features.append(informed_data)

        all_informed_data = torch.cat(all_informed_features, dim=-1) # -- (batch, 1, F)

        # -- self-supervised speech representations
        ssl_data = batch[self.config.ssl_features] # -- (batch, time, D)

        # -- cross-embed attention
        all_embed_informed_data = all_informed_data.repeat(1, ssl_data.shape[1], 1) # -- (batch, time, F)
        embed_mha_output = self.embed_mha(ssl_data, all_embed_informed_data, ssl_data, mask=None) # -- (batch, time, D), (batch, D, F)
        reduced_embed_mha_output = Reduce('b n d -> b d', 'mean')(embed_mha_output) # -- (batch, F)

        # -- cross-time attention
        all_time_informed_data = all_informed_data.repeat(1, ssl_data.shape[-1], 1) # -- (batch, D, F)
        time_mha_output = self.time_mha(ssl_data, all_time_informed_data, ssl_data, batch['mask_ssl']) # -- (batch, time, D), (batch, T, F)
        reduced_time_mha_output = Reduce('b n d -> b d', 'mean')(time_mha_output) # -- (batch, F)

        # mha_output = embed_mha_output + time_mha_output
        mha_output = torch.cat([reduced_embed_mha_output, reduced_time_mha_output], dim=-1).unsqueeze(1) # -- (batch, D*2)

        # -- classification
        logits = self.classifier(mha_output).squeeze(1)

        model_output['subject_id'] = batch['subject_id']
        model_output['sample_id'] = batch['sample_id']
        model_output['embeddings'] = mha_output
        model_output['logits'] = logits
        model_output['probs'] = torch.nn.functional.softmax(logits, dim = -1)
        model_output['preds'] = logits.argmax(dim = -1)
        model_output['labels'] = batch['label']
        model_output['loss'] = self.loss_criterion(logits, batch['label'])
        model_output[f'cross_embed_mha_scores'] = self.embed_mha.attn_scores
        model_output[f'cross_time_mha_scores'] = self.time_mha.attn_scores

        return model_output
