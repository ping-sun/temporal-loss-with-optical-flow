# video-segmentation-training

Training project for video segmentation

## Project structure:
``` sh
.
├── checkpoints
├── dataset                # dataset directory
├── dataloader             # dataloader
│   ├── data_generator.py
│   └── imagereader.py
├── net                    # net define
│   ├── lw_helpers.py
│   ├── lw_mobilenet.py
│   ├── lw_resnet.py
│   └── videoseg.py
├── trainval_video.py
└── utils                  # other utils
    ├── helper.py
    └── lr_scheduler.py
```

## Training:
### Prepare Data
  For consistency, you'd better link dataset to your local ./dataset directory:
  ``` sh
  ln -s path_to_dataset ./dataset/
  ```

### Run:
  ``` sh
  python3 trainval.py
  ```

## Use Dataloader:

  Data loader supports dataset list as input, example:
  ``` py
  mhp = imagefile('./dataset/LV-MHP-v2', 'list/train.txt')
  supervisely = imagefile('./dataset/Supervisely', 'train.txt',
                          img_dir='SuperviselyImages', label_dir='SuperviselyMasks')
  data_dataset = DataGenerator([mhp, supervisely], phase='val')
  data_loader = DataLoader(
      data_dataset, batch_size=8, shuffle=False, num_workers=8)
  ```

## Use Model:
  ``` py
  from net.lw_mobilenet import mbv2 as network
  net = network(2, pretrained=False) # pretrained=True will download the pretrained model on VOC
  ```

### Export to ONNX:
  ``` py
  x = torch.rand(1, 3, 512, 512)
  torch.onnx.export(net, x, "model.onnx", verbose=False)
  ```

### Convert ONNX to CoreML
  ``` sh
  convert-onnx-to-coreml model.onnx -o model
  ```
  Then `model.mlmodel` is the CoreML model.

