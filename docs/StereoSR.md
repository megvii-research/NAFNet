## NAFSSR: Stereo Image Super-Resolution Using NAFNet

The official pytorch implementation of the paper **[NAFSSR: Stereo Image Super-Resolution Using NAFNet](https://arxiv.org/abs/2204.08714)**.
You can get more infomation about NAFSSR with folloing links: [[video](https://drive.google.com/file/d/16w33zrb3UI0ZIhvvdTvGB2MP01j0zJve/view)]/[[slides](https://data.vision.ee.ethz.ch/cvl/ntire22/slides/Chu_NAFSSR_slides.pdf)]/[[poster](https://data.vision.ee.ethz.ch/cvl/ntire22/posters/Chu_NAFSSR_poster.pdf)].
#### Xiaojie Chu\*, Liangyu Chen\*, Wenqing Yu

>This paper proposes a simple baseline named NAFSSR for stereo image super-resolution. We use a stack of NAFNet's Block (NAFBlock) for intra-view feature extraction and combine it with Stereo Cross Attention Modules (SCAM) for cross-view feature interaction.

<img src=../figures/NAFSSR_arch.jpg>

>NAFSSR outperforms the state-of-the-art methods on the KITTI 2012, KITTI 2015, Middlebury, and Flickr1024 datasets. With NAFSSR, we won **1st place**  in the [NTIRE 2022 Stereo Image Super-resolution Challenge](https://codalab.lisn.upsaclay.fr/competitions/1598). 

<p align="center">
<img src=../figures/NAFSSR_params.jpg width=70%>
</p>




# Reproduce the Stereo SR Results 

## 1. Data Preparation
Follow previous works, our models are trained with Flickr1024 and Middlebury datasets, which is exactly the same as <a href="https://github.com/YingqianWang/iPASSR">iPASSR</a>. Please visit their homepage and follow their instructions to download and prepare the datasets.

#### Download and prepare the train set and place it in ```./datasets/StereoSR```

#### Download and prepare the evaluation data and place it in ```./datasets/StereoSR/test```

The structure of `datasets` directory should be like
```
    datasets
    ├── StereoSR
    │   ├── patches_x2
    │   │   ├── 000001
    │   │   ├── 000002
    │   │   ├── ...
    │   │   ├── 298142
    │   │   └── 298143
    │   ├── patches_x4
    │   │   ├── 000001
    │   │   ├── 000002
    │   │   ├── ...
    │   │   ├── 049019
    │   │   └── 049020
    |   ├── test
    │   |   ├── Flickr1024
    │   │   │   ├── hr
    │   │   │   ├── lr_x2
    │   │   │   └── lr_x4
    │   |   ├── KITTI2012
    │   │   │   ├── hr
    │   │   │   ├── lr_x2
    │   │   │   └── lr_x4
    │   |   ├── KITTI2015
    │   │   │   ├── hr
    │   │   │   ├── lr_x2
    │   │   │   └── lr_x4
    │   │   └── Middlebury
    │   │       ├── hr
    │   │       ├── lr_x2
    │   │       └── lr_x4
```

## 2. Evaluation


#### Download the pretrain model in ```./experiments/pretrained_models/```

| name | scale |#Params|PSNR|SSIM| pretrained models | configs |
|:----:|:----:|:----:|:----:|:----:|:----:|-----:|
|NAFSSR-T|x4|0.46M|23.69|0.7384|[gdrive](https://drive.google.com/file/d/1owfYG1KTXFMl4wHpUZefWAcVlBpLohe5/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1yC5XzJcL5peC1YuW3MkFMA?pwd=5j1u)|[train](../options/test/NAFSSR/NAFSSR-T_4x.yml) \| [test](../options/test/NAFSSR/NAFSSR-T_4x.yml)|
|NAFSSR-S|x4|1.56M|23.88|0.7468|[gdrive](https://drive.google.com/file/d/1RpfS2lemsgetIQwBwkZpZwLBJfOTDCU5/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1XvwM5KVhNsKAxWbxU85SFA?pwd=n5au)|[train](../options/test/NAFSSR/NAFSSR-S_4x.yml) \| [test](../options/test/NAFSSR/NAFSSR-S_4x.yml)|
|NAFSSR-B|x4|6.80M|24.07|0.7551|[gdrive](https://drive.google.com/file/d/1Su0OTp66_NsXUbqTAIi1msvsp0G5WVxp/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/18tVlH-QIVtvDC1LM2oPatw?pwd=3up5)|[train](../options/test/NAFSSR/NAFSSR-B_4x.yml) \| [test](../options/test/NAFSSR/NAFSSR-B_4x.yml)|
|NAFSSR-L|x4|23.83M|24.17|0.7589|[gdrive](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1P8ioEuI1gwydA2Avr3nUvw?pwd=qs7a)|[train](../options/test/NAFSSR/NAFSSR-L_4x.yml) \| [test](../options/test/NAFSSR/NAFSSR-L_4x.yml)|
|NAFSSR-T|x2|0.46M|28.94|0.9128|[gdrive](https://drive.google.com/file/d/1sBivtt5KaFMjMhBwyajYy1uemIEFQBvW/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1RDW923v0e0G_eYvTF8I7Dg?pwd=utgs)|[train](../options/test/NAFSSR/NAFSSR-T_2x.yml) \| [test](../options/test/NAFSSR/NAFSSR-T_2x.yml)|
|NAFSSR-S|x2|1.56M|29.19|0.9160|[gdrive](https://drive.google.com/file/d/1caVrp3fFSpwiU8RPGXe-zDD1tU110yxA/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1qJmQzv-YV1V9raim57pTIQ?pwd=t2or)|[train](../options/test/NAFSSR/NAFSSR-S_2x.yml) \| [test](../options/test/NAFSSR/NAFSSR-S_2x.yml)|
|NAFSSR-B|x2|6.80M|29.54|0.7551|[gdrive](https://drive.google.com/file/d/1gOfDTfyCaff_xNm86u8sN_3MtAZvnQEk/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1IkadqW1uWx9xM5K2ETJ9-A?pwd=pv1f)|[train](../options/test/NAFSSR/NAFSSR-B_2x.yml) \| [test](../options/test/NAFSSR/NAFSSR-B_2x.yml)|
|NAFSSR-L|x2|23.79M|29.68|0.9221|[gdrive](https://drive.google.com/file/d/1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf/view?usp=sharing)  \|  [baidu](https://pan.baidu.com/s/1GS6YQSSECH8hAKhvzw6GyQ?pwd=2v3v)|[train](../options/test/NAFSSR/NAFSSR-L_2x.yml) \| [test](../options/test/NAFSSR/NAFSSR-L_2x.yml)|

*PSNR/SSIM are evaluate on Flickr1024 test set.*


### Testing on KITTI2012, KITTI2015, Middlebury, Flickr1024 datasets

  * NAFSSR-T for 4x SR:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/NAFSSR/NAFSSR-T_x4.yml --launcher pytorch
```

  * NAFSSR-S for 4x SR:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/NAFSSR/NAFSSR-S_x4.yml --launcher pytorch
```

  * NAFSSR-B for 4x SR:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/NAFSSR/NAFSSR-B_x4.yml --launcher pytorch
```

  * NAFSSR-L for 4x SR:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/NAFSSR/NAFSSR-L_x4.yml --launcher pytorch
```

  * NAFSSR-L for 2x SR:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/NAFSSR/NAFSSR-L_x2.yml --launcher pytorch
```

* Test by a single gpu by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.




## 3. Training

* NAFNet-B for 4x SR:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/NAFSSR/NAFSSR-B_x4.yml --launcher pytorch
  ```

* NAFNet-S for 4x SR:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/NAFSSR/NAFSSR-S_x4.yml --launcher pytorch
  ```

* 8 gpus by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

  




## Citations
If NAFSSR helps your research or work, please consider citing NAFSSR.
```
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}
```
