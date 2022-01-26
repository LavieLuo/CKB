# Conditional Bures Metric for Domain Adaptation

This is the `Pytorch` implementation for **[Conditional Bures Metric for Domain Adaptation (CKB) (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.html)**.

## Overview

*"“Conditional Kernel Bures (CKB) is a conditional distribution adaptation model, which explores Wasserstein-Bures geometry and learns conditional invariant representations for knowledge transfer."*

### Insight
![CKB_Insight](https://github.com/LavieLuo/Datasets/blob/master/CKB_Insight.PNG)

### Network Architectures
![NetworkArchitectures](https://github.com/LavieLuo/Datasets/blob/master/CKB_Network.PNG)

## Environments
- Ubuntu 18.04
- python 3.6
- PyTorch 1.0

## Dataset
- The datasets should be placed in `./dataset`, e.g.,

  `./dataset/OfficeHome`

- The structure of the datasets should be like
```
OfficeHome (Dataset)
|- Art (Domain)
|  |- Alarm_Clock (Class)
|     |- XXXX.jpg (Sample) 
|     |- ...
|  |- Backpack (Class)
|  |- ...
|- Clipart (Domain)
|- Product (Domain)
|- Real_World (Domain)
```

## Train & Test
- For OfficeHome dataset with SGD or Adam optimizer, please run

  ``` 
  python main.py --dataset OfficeHome --exp_times 10 --batch_size 40 --CKB_lambda 1e-1 -- CKB_type hard --inv_epsilon 1e-2 --lr 1e-3 --optim_param GD
  python main.py --dataset OfficeHome --exp_times 10 --batch_size 40 --CKB_lambda 1e-1 -- CKB_type hard --inv_epsilon 1e-2 --lr 3e-4
  ```
  
  - For ImageCLEF dataset with SGD or Adam optimizer, please run

  ``` 
  python main.py --dataset ImageCLEF --exp_times 10 --batch_size 40 --CKB_lambda 1e0 --inv_epsilon 1e-1 --lr 1e-3 --optim_param GD
  python main.py --dataset ImageCLEF --exp_times 10 --batch_size 40 --CKB_lambda 1e0 --inv_epsilon 1e-1 --lr 3e-4
  ```

## Citation
If this repository is helpful for you, please cite our paper:
```
@inproceedings{luo2021conditional,
  title={Conditional Bures Metric for Domain Adaptation},
  author={Luo, You Wei and Ren, Chuan Xian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13989--13998},
  year={2021}
}
```

## Contact
If you have any questions, please feel free contact me via **luoyw28@mail2.sysu.edu.cn**.
