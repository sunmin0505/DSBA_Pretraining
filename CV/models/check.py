import timm

# for model in timm.list_models('*32*'):
#     print(model)

model = timm.create_model('vit_base_patch16_224', pretrained=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")