<br />
<p align="center" style="display: flex; flex-direction: column; align-items: center;">
  <h3 style="margin: 0;">
    <strong> A New Benchmark Model for Monocular 3D Lane Detection in Autonomous Driving using Transformative Architecture </strong>
  </h3>
  <span style="font-size: 16px; font-weight: 300; margin-top: 0;"> by Mohammed Fachry Dwi Handoko &amp; Prof. Ir. Handayani Tjandrasa, M.Sc. Ph.D. </span>
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


You can access the publication [here](http://repository.its.ac.id/id/eprint/117808).

![fig2](/assets/fig2.png)


## Environments
To set up the required packages, please refer to the [installation guide](./docs/install.md).

## Data
Please follow [data preparation](./docs/data_preparation.md) to download the dataset.

## Pretrained Models
Note that the performance of pretrained model is higher than the actual due to code refactoration and optimization. All models are uploaded to this specific [Google Drive](https://drive.google.com/drive/folders/1AhvLvE84vayzFxa0teRHYRdXz34ulzjB?usp=sharing).

| Dataset | Pretrained | Metrics | md5 |
| - | - | - | - |
| OpenLane-1000 | [Google Drive](https://drive.google.com/file/d/1jThvqnJ2cUaAuKdlTuRKjhLCH0Zq62A1/view?usp=sharing) | F1=0.6297 | d8ecb900c34fd23a9e7af840aff00843 |
| OpenLane-1000 (Lite version) | [Google Drive](https://drive.google.com/file/d/1WD5dxa6SI2oR9popw3kO2-7eGM2z-IHY/view?usp=sharing) | F1=0.6212 | 918de41d0d31dbfbecff30

## Evaluation
You can download the [pretrained models](#pretrained-models) to `./pretrained_models` directory and refer to the [eval guide](./docs/train_eval.md#evaluation) for evaluation.

## Train
Please follow the steps in [training](./docs/train_eval.md#train) to train the model.

## Benchmark

### OpenLane
| Models | F1 | Accuracy | X error <br> near \| far | Z-error <br> near \| far |
| ----- | -- | -------- | ------- | ------- |
| **Thesis-LATR** | **55.6** | **91.9** | **0.347** \| **0.446** | **0.529** \| **0.626** |
| LATR | 61.9 | 92.0 | 0.219 \| 0.259 | 0.075 \| 0.104 |
| 3DLaneNet | 44.1 | - | 0.479 \| 0.572 | 0.367 \| 0.443 |
| GenLaneNet | 32.3 | - | 0.593 \| 0.494 | 0.140 \| 0.195 |
| PersFormer | 50.5 | 89.5 | 0.319 \| 0.325 | 0.112 \| 0.141 |
| PersFormer-Res50 | 53.0 | 89.2 | 0.321 \| 0.303 | 0.085 \| 0.118 |

## Acknowledgment
This library is inspired by [LATR](https://github.com/JMoonr/LATR), [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), and other related works. We thank them ever so kindly for sharing the codes, papers, and datasets to be used as references in this project.


## Citation
If you find that this project is useful for your research, please consider citing the paper:

```tex
@article{handoko2025thesis,
  title={A New Benchmark Model for Monocular 3D Lane Detection in Autonomous Driving using Transformative Architecture},
  author={Handoko, Mohammed Fachry Dwi and Tjandrasa, Handayani},
  journal={ITS Repository preprint uri:http://repository.its.ac.id/id/eprint/117808},
  year={2025}
}
```
