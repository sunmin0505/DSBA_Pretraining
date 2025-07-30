import timm

for model in timm.list_models('*vit*'):
    print(model)