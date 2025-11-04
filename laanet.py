import sys

sys.path.append("repos/LAA-Net")

from models.builder import build_model, MODELS


cfg = dict(
    type="PoseEfficientNet",
    model_name="efficientnet-b4",
    num_layers="B4",
    include_top=False,
    include_hm_decoder=True,
    head_conv=64,
    use_c2=False,
    use_c3=True,
    use_c4=True,
    use_c51=True,
    efpn=True,
    tfpn=False,
    se_layer=False,
    heads=dict(hm=1, cls=1, cstency=256),
    INIT_WEIGHTS=dict(pretrained=True, advprop=True),
)
model = build_model(cfg, MODELS)
model.init_weights(pretrained=True)
model.eval()

ckpt_path = "repos/LAA-Net/pretrained/PoseEfficientNet_EFN_hm100_EFPN_NoBasedCLS_Focal_C3_256Cstency100_32BI_SAM(Adam)_ADV_Erasing1_OutSigmoid_model_best.pth"

import torch

checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

print("Model loaded successfully.")
print(model)


"""
MODEL:
  type: PoseEfficientNet
  model_name: efficientnet-b4
  num_layers: B4
  include_top: False
  include_hm_decoder: True
  head_conv: 64
  use_c2: False
  use_c3: True
  use_c4: True
  use_c51: True
  efpn: True
  tfpn: False
  se_layer: False
  heads:
    hm: 1
    cls: 1
    cstency: 256
  INIT_WEIGHTS:
    pretrained: True
    advprop: True
"""
