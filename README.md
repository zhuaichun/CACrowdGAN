# CACrowdGAN
CACrowdGAN: Cascaded Attentional Generative Adversarial Network for Crowd Counting


## Dependencies

- Python>=3.0
- opencv3
- numpy
- pytorch >= 1.3.0
- Image
- h5py

## Dataset
The generation of Ground Truth can refer to [**CSRNet**](https://github.com/leeyeehoo/CSRNet-pytorch)


## Weights
We provide the best results of the two models on Shanghai A and Shanghai B respectively. The names are defined as **[model name] + \_model\_ + [dataset name] + .pth.tar**.


## evaluation
```
# in test.py
val_data_path = '../part_A/test_data/images/'
val_list = []
for i in os.listdir(val_data_path):
    val_list.append(val_data_path + i)

weight_path = './CSRNet_model_bestB.pth.tar'
density_model = False
if density_model:
    model = Dense()
else:
    model = CSRNet()
```
- val_data_path : The path of the dataset.
- weight_path  : The path of the weight.
- density_model  : The flag to choose the used model. If True, Dense model will be used to predict the results.

After setting the corresponding configuration, running the following commands could evaluate the model.
```
python3 test.py
```


