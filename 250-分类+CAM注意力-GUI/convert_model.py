import torch
from models.mobilenetv1 import MobileNet
from torchvision.models import mobilenet_v2,resnet18
from torch.utils.mobile_optimizer import optimize_for_mobile

model = MobileNet(num_classes=3)

# model.eval()
# input_tensor = torch.rand(1,3,224,224)

model.load_state_dict(torch.load("./weights/model-epoch-26-acc-0.9914089347079038.pth"))#保存的训练模型
model.eval()#切换到eval（）
example = torch.rand(1, 3, 224, 224)#生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")

# dummy_input = torch.rand(1, 3, 224, 224)
# torchscript_model = torch.jit.trace(model, input_tensor)
#
#
# torchscript_model_optimized = optimize_for_mobile(torchscript_model)
# torch.jit.save(torchscript_model_optimized, "mobilenetv2_quantized.pt")