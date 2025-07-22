# Learning In-Hand Translation Using Tactile Skin With Shear and Normal Force Sensing

This is the code release of the following paper:<br>
<b>Learning In-Hand Translation Using Tactile Skin With Shear and Normal Force Sensing</b><br>
[Jessica Yin](https://jessicayin.com/),
[Haozhi Qi](https://haozhi.io/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/),
[James Pikul](https://pikulgroup.engr.wisc.edu/),
[Mark Yim](https://www.modlabupenn.org/),
[Tess Hellebrekers](http://www.tesshellebrekers.com/)<br>
[ArXiv](https://arxiv.org/abs/2407.07885), [Project Website](https://jessicayin.github.io/tactile-skin-rl/)

## Overview

This repository contains the following functionalities:
1. We offer a standalone implementation for the tactile skin model implementation, for use in your own pipeline.
2. We provide the pretrained in-hand translation policy in simulation with IsaacGym Preview 4.0 ([Download](https://drive.google.com/file/d/1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9)). Model weights can be downloaded [here](https://drive.google.com/drive/folders/1BMPx382AQrugZmAXYDswY-Dge9GrpZa0?usp=sharing).

Please refer to the [Hora](https://github.com/haozhiqi/hora) repo for the RL training pipeline implementation, we primarily show the tactile skin model and how it is integrated here. Additionally, please refer to the Hora repo for policy deployment code.

TODO: Instructions for policy visualization in IsaacGym.

## Acknowledgement

This repository is based on [Hora](https://github.com/haozhiqi/hora) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs). 

## Reference
If you find the paper or this codebase helpful in your research, please consider citing:

```
@article{yin2024learninginhandtranslation,
      title={Learning In-Hand Translation Using Tactile Skin With Shear and Normal Force Sensing}, 
      author={Jessica Yin and Haozhi Qi and Jitendra Malik and James Pikul and Mark Yim and Tess Hellebrekers},
      year={2024},
      eprint={2407.07885},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.07885}, 
}
```
