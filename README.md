# CACrowdGAN
CACrowdGAN: Cascaded Attentional Generative Adversarial Network for Crowd Counting


## Dependencies

- Python>=3.0
- opencv3
- numpy
- pytorch >= 1.3.0
- Image
- h5py
## What is CACrowdGAN?
It contains two components: the attentional generator and the cascaded attentional discriminator. The attentional generator has an attention module and a density
module. The attention module is developed to provide the attentional input of the density module, while the density module is designed to generate density maps. In addition,
a novel cascaded attentional discriminator is proposed to synthesize attentional-driven fine-grained details at different crowd regions of the input image. The proposed discriminator module is built by an Hourglass-based structure (can be seen as an auto-encoder) which enables the discriminator to be used in a cascaded form and simultaneously the more precise per-pixel reconstruction loss. 
![image](https://github.com/xxx/xxx/blob/master/xxx/xxx.png)

## Dataset
ShanghaiTech Dataset: [**Google Drive**](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)


##  Models
We provide the best results of the two models on Shanghai A and Shanghai B respectively. The names are defined as **[model name] + \_model\_ + [dataset name] + .pth.tar**.


## Running
```
# in run.py
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
python3 run.py
```


