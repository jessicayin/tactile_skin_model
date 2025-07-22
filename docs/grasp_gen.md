Both the oracle and tactile policies assume that the object is initialized in a stable grasp, similar to [Hora](https://github.com/haozhiqi/hora). We pre-generate caches of stable grasp poses for training. 

First, we define canonical poses for the fingers, which varies for each scale of the object. Because we train for a large range of scales, the same canonical pose will not work for the entire range. Then, we define a range of relative offsets within $[-0.25, 0.25]$ rad to randomly sample from. Additionally, we define a canonical position for the object and sample from a relative offset of $[-10, 10]$ cm along the x-axis for the object. The canonical object pose is the same for all scales: $[x, y, z, w_{\text{quat}}, x_{\text{quat}}, y_{\text{quat}}, z_\text{{quat}}] = [0, 0, 0.52, 0, -0.5, 0, 0.5]$. 

We let the simulation run for 0.5 s. The finger and object positions are saved (i.e., deemed a stable grasp) if the following conditions are satisfied:

1. All fingertips are within 10 cm of the object.
2. At least 2 fingers are in contact with the object.
3. The object did not fall below the hand.
4. The object is in contact with the palm.

We sampled 50,000 grasps and object poses for each object scale from 70%-120% of the canonical cylinder. 

To run the grasp generation script:

```./scripts/gen_grasp.sh 0 1```

The first arg is the GPU device number and the second arg is the object scale, where 1 = 100% = canonical scale.

If you want to visualize the grasp generation, please set `headless=False` in `gen_grasp.sh` and decrease the number of envs so that the computer can render it. Additionally, there may be an error with CUDA, in which case please comment out GPU arg input and `CUDA_VISIBLE_DEVICES` in the `gen_grasp.sh`.