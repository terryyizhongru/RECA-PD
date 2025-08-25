import torch
from einops.layers.torch import Reduce
from .multiheaded_attention import MultiHeadedAttention

class SelfInfModel(torch.nn.Module):
    """Informed Self-Attention Baseline Model.
    """

    def __init__(self, config):
        super(SelfInfModel, self).__init__()

        self.config = config
        self.attn_type = self.config.model

        # -- computing informed-based speech features input dimension
        informed_input_dim = 0
        for feature in self.config.features:
            informed_input_dim += feature['input_dim']

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(
                informed_input_dim,
                self.config.model_conf['latent_dim'],
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.config.model_conf['dropout']),
        )

        # -- model architecture setup
        self.query_dim = self.config.model_conf['latent_dim']
        self.key_dim = self.config.model_conf['latent_dim']
        self.value_dim = self.config.model_conf['latent_dim']

        self.mha = MultiHeadedAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_heads=self.config.model_conf['num_heads'],
            dropout_rate=self.config.model_conf['dropout'],
            attn_type=self.attn_type,
        )

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

        # -- concatenating all informed-based speech features
        all_informed_features = []
        for feature in self.config.features:
            feature_id = feature['name']
            informed_data = batch[feature_id] # -- (batch, 1, input_dim)
            all_informed_features.append(informed_data)
        all_informed_data = torch.cat(all_informed_features, dim=-1) # -- (batch, 1, F)

        # -- projection embedding
        all_informed_data = self.projection(all_informed_data)

        # -- self-attention mechanism
        mha_output = self.mha(all_informed_data, all_informed_data, all_informed_data, mask=None) # -- (batch, 1, F)

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
        model_output[f'self_inf_mha_scores'] = self.mha.attn_scores

        return model_output


class SelfSSLModel(torch.nn.Module):
    """SSL Self-Attention Baseline Model.
    """

    def __init__(self, config):
        super(SelfSSLModel, self).__init__()

        self.config = config
        self.attn_type = self.config.model

        if self.config.ssl_features_conf['input_dim'] == self.config.model_conf['latent_dim']:
            self.linear_projection = None
        else:
            self.linear_projection = torch.nn.Linear(
                self.config.ssl_features_conf['input_dim'],
                self.config.model_conf['latent_dim'],
                bias=False,
            )

        # -- model architecture setup
        # [TxD] · [DxT] --> softmax([TxT]) · [TxD] --> [TxD] --> [1xD] -- Classification
        self.query_dim = self.config.model_conf['latent_dim']
        self.key_dim = self.config.model_conf['latent_dim']
        self.value_dim = self.config.model_conf['latent_dim']

        # -- attention mechanism
        self.mha = MultiHeadedAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_heads=self.config.model_conf['num_heads'],
            dropout_rate=self.config.model_conf['dropout'],
            attn_type=self.attn_type,
        )

        # -- classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.key_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.config.model_conf['latent_dim'],
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

        # -- self-supervised speech representations
        ssl_data = batch[self.config.ssl_features] # -- (batch, time, D)

        if self.linear_projection is not None:
            ssl_data = self.linear_projection(ssl_data)

        # -- self-attention mechanism
        mha_output = self.mha(ssl_data, ssl_data, ssl_data, mask=batch['mask_ssl']) # -- (batch, T, D), (batch, T, T)
        reduced_mha_output = Reduce('b n d -> b d', 'mean')(mha_output).unsqueeze(1) # -- (batch, D)

        # -- classification
        logits = self.classifier(reduced_mha_output).squeeze(1)

        model_output['subject_id'] = batch['subject_id']
        model_output['sample_id'] = batch['sample_id']
        model_output['embeddings'] = reduced_mha_output
        model_output['logits'] = logits
        model_output['probs'] = torch.nn.functional.softmax(logits, dim = -1)
        model_output['preds'] = logits.argmax(dim = -1)
        model_output['labels'] = batch['label']
        model_output['loss'] = self.loss_criterion(logits, batch['label'])
        model_output[f'self_ssl_mha_scores'] = self.mha.attn_scores

        return model_output
