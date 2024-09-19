import torch
import os

import torch.nn.functional as F
from collections import OrderedDict

from network.base_model import BaseModel
from copy import deepcopy
from utils import load_ckpt
from .mobilenetv2 import mobilenetv2


class Model(BaseModel):
    def __init__(self, opt, config):
        super(Model, self).__init__(opt, config)
        self.opt = opt
        self.config = config

        backbone_config = deepcopy(config.model.backbone._dict)
        arch = backbone_config.pop('type')
        pretrained = backbone_config.pop('pretrained')
        softlabel = config.data.soft_label
        class_names = config.data.class_names
        if softlabel:
            num_classes = 2
        else:
            num_classes = len(class_names) + 1

        backbone_config['num_classes'] = num_classes 

        if arch == 'MobileNetV2':
            self._classifier = mobilenetv2(**backbone_config)
            if pretrained is not None:
                state_dict = torch.load(pretrained, map_location='cpu')
                load_ckpt(self._classifier, state_dict)
        else:
            raise NotImplementedError(f'arch "{arch}" not implemented error.')

        self.init_common()

    def update(self, input, label):
        predicted = self.classifier(input)
        loss = self.loss(predicted, label, avg_meters=self.avg_meters)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None
    
    def export_onnx(self, export_path):
        from misc_utils import color_print
        export_scale = self.config.data.export.scale
        w, h = export_scale
        inputx = torch.randn([1, 3, h, w])
        color_print(f'export onnx to "{export_path}"', 3)
        torch.onnx.export(
            self.classifier, inputx, export_path,
            opset_version=11,
        )

    def check_onnx(self, onnx_path):
        import onnx
        import onnxruntime as rt
        import numpy as np
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        output_names = [node.name for node in onnx_model.graph.output]
        print(f'output_names(classifier): {output_names}')
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)

        export_scale = self.config.data.export.scale
        w, h = export_scale
        tensor_input = torch.ones((1, 3, h, w)) * 0.5
        # cls_score, bbox_pred = pytorch_model(tensor_input)
        preds = self.classifier(tensor_input)
        sess = rt.InferenceSession(onnx_path)
        ox_rets = sess.run(
            None,
            {net_feed_input[0]: tensor_input.detach().numpy()},
        )
        pytorch_rets_list = []
        pytorch_rets_list.append(preds)
        
        #pytorch_rets_list.append(cls_score)
        #pytorch_rets_list.append(bbox_pred)

        for o_res, p_res in zip(ox_rets, pytorch_rets_list):
            assert o_res.shape == p_res.shape
            p_res = p_res.cpu().detach().numpy()
            print('max diff:', np.max(np.abs(o_res-p_res)))
            # np.testing.assert_allclose(o_res, p_res, rtol=1e-03, atol=1e-05,)