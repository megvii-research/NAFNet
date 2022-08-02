# reproduce the SIDD dataset results 



### 1. Data Preparation

##### Download the train set and place it in ```./datasets/SIDD/Data```:

* [google drive](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1EnBVjrfFBiXIRPBgjFrifg?pwd=sl6h), 
* ```python scripts/data_preparation/sidd.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/SIDD/val/```:

  * [google drive](https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1I9N5fDa4SNP0nuHEy6k-rw?pwd=59d7), 
  * it should be like ```./datasets/SIDD/val/input_crops.lmdb``` and ```./datasets/SIDD/val/gt_crops.lmdb```



### 2. Training

* NAFNet-SIDD-width32:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/NAFNet-width32.yml --launcher pytorch
  ```

* NAFNet-SIDD-width64:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/NAFNet-width64.yml --launcher pytorch
  ```
  
* Baseline-SIDD-width32:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/Baseline-width32.yml --launcher pytorch
  ```

* Baseline-SIDD-width64:

  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/Baseline-width64.yml --launcher pytorch
  ```

* 8 gpus by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

  


### 3. Evaluation


##### Download the pretrain model in ```./experiments/pretrained_models/```

  * **NAFNet-SIDD-width32**: [google drive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1Xses38SWl-7wuyuhaGNhaw?pwd=um97)

  * **NAFNet-SIDD-width64**: [google drive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/198kYyVSrY_xZF0jGv9U0sQ?pwd=dton)

  * **Baseline-SIDD-width32**: [google drive](https://drive.google.com/file/d/1NhqVcqkDcYvYgF_P4BOOfo9tuTcKDuhW/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1wkskmCRKhXq6dGa6Ns8D0A?pwd=0rin)

  * **Baseline-SIDD-width64**: [google drive](https://drive.google.com/file/d/1wQ1HHHPhSp70_ledMBZhDhIGjZQs16wO/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1ivruGfSRGfWq5AEB8qc7YQ?pwd=t9w8)
    

##### Testing on SIDD dataset	

  * NAFNet-SIDD-width32:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/NAFNet-width32.yml --launcher pytorch
```

  * NAFNet-SIDD-width64:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/NAFNet-width64.yml --launcher pytorch
```

  * Baseline-SIDD-width32:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/Baseline-width32.yml --launcher pytorch
```

  * Baseline-SIDD-width64:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt ./options/test/SIDD/Baseline-width64.yml --launcher pytorch
```

* Test by a single gpu by default. Set ```--nproc_per_node``` to # of gpus for distributed validation.

