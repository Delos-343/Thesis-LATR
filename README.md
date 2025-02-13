<br />
<p align="center">
  <h3 align="center"><strong> A New Benchmark Model for Monocular 3D Lane Detection in Autonomous Driving using Transformative Architecture </strong></h3>
</p>

<p align="center">
  <a href="http://repository.its.ac.id/id/eprint/117808" target='_blank'>
    <!-- <img src="https://img.shields.io/badge/arXiv-%F0%9F%93%83-yellow"> -->
    <img src="https://img.shields.io/badge/arXiv-2308.04583-b31b1b.svg">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Delos-343.Thesis-LATR&left_color=gray&right_color=yellow">
  </a>
    <a href="https://github.com/Delos-343/Thesis-LATR target='_blank'>
     <img src="https://img.shields.io/github/stars/Delos-343/Thesis-LATR?style=social">
  </a>
</p>


This is the implementation of [A New Benchmark Model for Monocular 3D Lane Detection in Autonomous Driving using Transformative Architecture](http://repository.its.ac.id/id/eprint/117808).

![fig2](/assets/fig2.png)


## Environments
To set up the required packages, please refer to the [installation guide](./docs/install.md).

## Data
Please follow [data preparation](./docs/data_preparation.md) to download dataset.

## Pretrained Models
Note that the performance of pretrained model is higher than the actual due to code refactoration and optimization. All models are uploaded to [Google Drive](https://drive.google.com/drive/folders/1AhvLvE84vayzFxa0teRHYRdXz34ulzjB?usp=sharing).

| Dataset | Pretrained | Metrics | md5 |
| - | - | - | - |
| OpenLane-1000 | [Google Drive](https://drive.google.com/file/d/1jThvqnJ2cUaAuKdlTuRKjhLCH0Zq62A1/view?usp=sharing) | F1=0.6297 | d8ecb900c34fd23a9e7af840aff00843 |
| OpenLane-1000 (Lite version) | [Google Drive](https://drive.google.com/file/d/1WD5dxa6SI2oR9popw3kO2-7eGM2z-IHY/view?usp=sharing) | F1=0.6212 | 918de41d0d31dbfbecff3001c49dc296 |
| ONCE | [Google Drive](https://drive.google.com/file/d/12kXkJ9tDxm13CyFbB1ddt82lJZkYEicd/view?usp=sharing) | F1=0.8125 | 65a6958c162e3c7be0960bceb3f54650 |
| Apollo-balance | [Google Drive](https://drive.google.com/file/d/1hGyNrYi3wAQaKbC1mD_18NG35gdmMUiM/view?usp=sharing) | F1=0.9697 | 551967e8654a8a522bdb0756d74dd1a2 |
| Apollo-rare | [Google Drive](https://drive.google.com/file/d/19VVBaWBnWiEqGx1zJaeXF_1CKn88G5v0/view?usp=sharing) | F1=0.9641 | 184cfff1d3097a9009011f79f4594138 |
| Apollo-visual | [Google Drive](https://drive.google.com/file/d/1ZzaUODYK2dyiG_2bDXe5tiutxNvc71M2/view?usp=sharing) | F1=0.9611 | cec4aa567c264c84808f3c32f5aace82 |


## Evaluation
You can download the [pretrained models](#pretrained-models) to `./pretrained_models` directory and refer to the [eval guide](./docs/train_eval.md#evaluation) for evaluation.

## Train
Please follow the steps in [training](./docs/train_eval.md#train) to train the model.

## Benchmark

### OpenLane
| Models | F1 | Accuracy | X error <br> near \| far | Z-error <br> near \| far |
| ----- | -- | -------- | ------- | ------- |
| 3DLaneNet | 44.1 | - | 0.479 \| 0.572 | 0.367 \| 0.443 |
| GenLaneNet | 32.3 | - | 0.593 \| 0.494 | 0.140 \| 0.195 |
| Cond-IPM | 36.3 | - | 0.563 \| 1.080 | 0.421 \| 0.892 |
| PersFormer | 50.5 | 89.5 | 0.319 \| 0.325 | 0.112 \| 0.141 |
| CurveFormer | 50.5 | - | 0.340 \| 0.772 | 0.207 \| 0.651 |
| PersFormer-Res50 | 53.0 | 89.2 | 0.321 \| 0.303 | 0.085 \| 0.118 |
| **LATR** | 61.9 | 92.0 | 0.219 \| 0.259 | 0.075 \| 0.104 |
| **Thesis-LATR** | 55.6 | 91.9 | 0.347 \| 0.446 | 0.529 \| 0.626 |

## Acknowledgment
This library is inspired by [LATR](https://github.com/JMoonr/LATR), [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), and other related works. We thank them kindly for sharing the code and datasets.


## Citation
If you find that this project is useful for your research, please consider citing the paper:

```tex
@article{luo2023latr,
  title={A New Benchmark Model for Monocular 3D Lane Detection in Autonomous Driving using Transformative Architecture},
  author={Handoko, Mohammed Fachry Dwi and Tjandrasa, Handayani},
  journal={ITS Repository preprint uri:http://repository.its.ac.id/id/eprint/117808},
  year={2025}
}
```
