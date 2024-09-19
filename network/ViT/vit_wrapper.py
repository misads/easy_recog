import torch.nn as nn

from .vit import Transformer

class Classifier(nn.Module):
    def __init__(self, opt, config, zero_head=False, vis=False):
        super(Classifier, self).__init__()
        self.num_classes = len(config.data.class_names)
        self.zero_head = zero_head

        model_config = config.model.backbone
        img_size = model_config.img_size
        hidden_size = model_config.hidden_size
        # transformer_config = model_config.transformer

        self.transformer = Transformer(model_config, img_size, vis)
        self.head = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        return logits

        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        #     return loss
        # else:
        #     return logits, attn_weights