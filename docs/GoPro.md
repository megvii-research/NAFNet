# reproduce the GoPro dataset results 



### 1. Data Preparation

##### Download the train set and place it in ```./datasets/GoPro/train```:

* [google drive](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1fdsn-M5JhxCL7oThEgt1Sw?pwd=9d26)
* it should be like ```./datasets/GoPro/train/input ``` and ```./datasets/GoPro/train/target```
* ```python scripts/data_preparation/gopro.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/GoPro/test/```:

  * [google drive](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1oZtEtYB7-2p3fCIspky_mw?pwd=rmv9)
  * it should be like ```./datasets/GoPro/test/input.lmdb``` and ```./datasets/GoPro/test/target.lmdb```



### 2. Training

* NAFNet-GoPro-width32:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/NAFNet-width32.yml --launcher pytorch
  ```

* NAFNet-GoPro-width64:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/NAFNet-width64.yml --launcher pytorch
  ```

* Baseline-GoPro-width32:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/Baseline-width32.yml --launcher pytorch
  ```
  
* Baseline-GoPro-width64:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/Baseline-width64.yml --launcher pytorch
  ```
  
* 8 gpus by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

  


### 3. Evaluation


##### Download the pretrain model in ```./experiments/pretrained_models/```
  * **NAFNet-GoPro-width32**: [google drive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1AbgG0yoROHmrRQN7dgzDvQ?pwd=so6v)
  * **NAFNet-GoPro-width64**: [google drive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1g-E1x6En-PbYXm94JfI1vg?pwd=wnwh)
  * **Baseline-GoPro-width32**: [google drive](https://drive.google.com/file/d/14z7CxRzVkYEhFgsZg79GlPTEr3VFIGyl/view?usp=sharing)  or [百度网盘](https://pan.baidu.com/s/1WnFKYTAQyAQ9XuD5nlHw_Q?pwd=oieh)
  * **Baseline-GoPro-width64**: [google drive](https://drive.google.com/file/d/1yy0oPNJjJxfaEmO0pfPW_TpeoCotYkuO/view?usp=sharing)  or [百度网盘](https://pan.baidu.com/s/1Fqi2T4nyF_wo4wh1QpgIGg?pwd=we36)



##### Testing on GoPro dataset	

  * NAFNet-GoPro-width32:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/NAFNet-width32.yml --launcher pytorch
```

  * NAFNet-GoPro-width64:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/NAFNet-width64.yml --launcher pytorch
```

  * Baseline-GoPro-width32:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/Baseline-width32.yml --launcher pytorch
```

  * Baseline-GoPro-width64:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/GoPro/Baseline-width64.yml --launcher pytorch
```

* Test by a single gpu by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

