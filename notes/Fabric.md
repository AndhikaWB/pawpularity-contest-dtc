## Intro

Fabric is a stripped down version of PyTorch Lightning. It doesn't force you to use their `Trainer` or `LightningModule`, so your PyTorch code will still look vanilla. I prefer Fabric because of these reasons:
- Most of the times, I just want to move everything to GPU (PyTorch doesn't have an easy way to do this). I don't actually care about other Lightning features, except maybe the `EarlyStopping` and a few other callbacks.
- I hate how `LightningModule` bundles training steps together with the model itself. In my opinion, these two things should be separate. I should be able to do whatever I want to my model later, why bundle it with the model? A model shouldn't be tied to just one training method!
- Compatibility, especially with other libraries like MLFlow. Sure, Lightning has a built-in MLFlow logger, but this thing is automatic and sometimes you want a precise control over it (what to log and when). What happen if there's a major update? Surely I don't want Lightning to break things or hold me back.
- Sometimes, a simple, long training loop is better for observability. Lightning breaks the training loop into multiple steps by default, and debugging which things execute on which step can be tiring. Not to mention the pain if you want to override their behavior, might as well write these things from scratch.

## Behavior

You setup Fabric using something like this:

```python
fabric = Fabric(accelerator = 'gpu')
model, optimizer = fabric.setup(model, optimizer)
train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
```

Internally, Fabric will call `to_device` to move every tensor on these objects to GPU (amongst other things like DDP support and custom float precision if you set them up). To achieve this, Fabric will wrap these objects into Fabric classes. If you use `Adam` as the optimizer, the class type now become `FabricAdam`, and if you use `torch.nn.Module` the model type will now become `_FabricModule`. I assume something similar for the `DataLoader`.

These Fabric classes are just dummy classes, they behave just like the original PyTorch classes, but of course you don't want to pickle/save these objects, you should unwrap them before. As of now, there's no universal function you can call to unwrap these classes, but you can access the original class by using either `.module` or `.optimizer` attribute.

What happen if I trained the Fabric model, will the weight of the Fabric model match the unwrapped (original) model? Well, I checked by comparing `model.state_dict()` and `model.module.state_dict()`, and they seem to be the same. However, if I call `str(model)` and `str(model.module)`, they will still have different result.

<details>
  <summary>Result</summary>

`print(str(model))`:

```
_FabricModule(
  (_forward_module): PawModel(
    (img_input): Sequential(
      (0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Linear(in_features=8192, out_features=128, bias=True)
    )
    (feat_input): Linear(in_features=12, out_features=128, bias=True)
    (comb_input): Sequential(
      (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=256, out_features=1, bias=True)
      (2): Sigmoid()
    )
  )
  (_original_module): PawModel(
    (img_input): Sequential(
      (0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Linear(in_features=8192, out_features=128, bias=True)
    )
    (feat_input): Linear(in_features=12, out_features=128, bias=True)
    (comb_input): Sequential(
      (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=256, out_features=1, bias=True)
      (2): Sigmoid()
    )
  )
)
```

`print(str(model.module))`:

```
PawModel(
  (img_input): Sequential(
    (0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Flatten(start_dim=1, end_dim=-1)
    (10): Linear(in_features=8192, out_features=128, bias=True)
  )
  (feat_input): Linear(in_features=12, out_features=128, bias=True)
  (comb_input): Sequential(
    (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Linear(in_features=256, out_features=1, bias=True)
    (2): Sigmoid()
  )
)
```

</details>

So just to be safe, just use the original model by calling `model.module` instead of `model` directly, especially if you want to save the model information and use them again later.