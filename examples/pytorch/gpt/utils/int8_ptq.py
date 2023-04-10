import torch

def quantize(weight, quantization_bit_width):
    # import ipdb; ipdb.set_trace()
    weight_scale = (weight.abs().max(dim=-1).values / (2 ** (quantization_bit_width - 1)))
    weight = torch.round(weight / weight_scale[:, None]).to(torch.int8)
    weight = torch.clip(weight, -128, 127)
    return weight.cpu(), weight_scale.cpu()

ckpt_path="pytorch_model.bin"
model = torch.load(ckpt_path)

q_model = {}
scale_dict = {}
for name, param in model.items():
    if 'proj.weight' in name or 'fc1.weight' in name or 'fc2.weight' in name:
        q_param, scale = quantize(param.cuda(), 4)
        model[name] = q_param
        module_name = '.'.join(name.split('.')[:-1])
        scale_dict[module_name] = {'scale':scale}
q_model['model'] = model
q_model['quant_factors'] = scale_dict

torch.save(q_model, 'opt-125m-w8-bin.pt')
    