# PointNetGPD: Detecting Grasp Configurations from Point Sets

# To run inference (main_test) you need to check the input size. It must match the input size that the network was trained on. 
Currently all main_ have a default of 750 input points. 

PointNetGPD (ICRA 2019, [arXiv](https://arxiv.org/abs/1809.06267)) is an end-to-end grasp evaluation model to address the challenging problem of localizing robot grasp configurations directly from the point cloud.

PointNetGPD is light-weighted and can directly process the 3D point cloud that locates within the gripper for grasp evaluation. Taking the raw point cloud as input, our proposed grasp evaluation network can capture the complex geometric structure of the contact area between the gripper and the object even if the point cloud is very sparse.

To further improve our proposed model, we generate a larger-scale grasp dataset with 350k real point cloud and grasps with the [YCB objects Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) for training.

<img src="data/grasp_pipeline.svg" width="100%" alt="grasp_pipeline">

## Video

[![Video for PointNetGPD](https://img.youtube.com/vi/RBFFCLiWhRw/0.jpg )](https://www.youtube.com/watch?v=RBFFCLiWhRw)
## Before Install
- All the code should be installed in the following directory:
```
mkdir -p $HOME/code/
cd $HOME/code/
```

- Set environment variable `PointNetGPD_FOLDER` in your `$HOME/.bashrc` file.
```
export PointNetGPD_FOLDER=$HOME/code/PointNetGPD
```

## Install all the requirements (Using a virtual environment is recommended)
1. Install `pcl-tools` via `sudo apt install pcl-tools`.
1. An example for create a virtual environment: `conda create -n pointnetgpd python=3.10 numpy ipython matplotlib opencv mayavi -c conda-forge`
1. Make sure in your Python environment do not have same package named ```meshpy``` or ```dexnet```.
1. Install PyTorch: https://pytorch.org/get-started/locally/
1. Clone this repository:
    ```bash
    cd $HOME/code
    git clone https://github.com/lianghongzhuo/PointNetGPD.git
    ```

1. Install our requirements in `requirements.txt`
    ```bash
    cd $PointNetGPD_FOLDER
    pip install -r requirements.txt
    ```
1. Install our modified meshpy (Modify from [Berkeley Automation Lab: meshpy](https://github.com/BerkeleyAutomation/meshpy))
    ```bash
    cd $PointNetGPD_FOLDER/meshpy
    python setup.py develop
    ```

1. Install our modified dex-net (Modify from [Berkeley Automation Lab: dex-net](https://github.com/BerkeleyAutomation/dex-net))
    ```bash
    cd $PointNetGPD_FOLDER/dex-net
    python setup.py develop
    ```
1. Modify the gripper configurations to your own gripper
    ```bash
    vim $PointNetGPD_FOLDER/dex-net/data/grippers/robotiq_85/params.json
    ```
    These parameters are used for dataset generation:
    ```bash
    "min_width":
    "force_limit":
    "max_width":
    "finger_radius":
    "max_depth":
    ```
    These parameters are used for grasp pose generation at experiment:
    ```bash
    "finger_width":
    "real_finger_width":
    "hand_height":
    "hand_height_two_finger_side":
    "hand_outer_diameter":
    "hand_depth":
    "real_hand_depth":
    "init_bite":
    ```
![](data/gripper.svg)

## Generated Grasp Dataset Download
- You can download the dataset from: https://tams.informatik.uni-hamburg.de/research/datasets/PointNetGPD_grasps_dataset.zip
- After download, extract the zip file and rename it to `ycb_grasp`
- Move the folder to `$PointNetGPD_FOLDER/PointNetGPD/data/`

## Generate Your Own Grasp Dataset

1. Download YCB object set from [YCB Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/).
A command line tool for download ycb dataset can be found at: [ycb-tools](https://github.com/lianghongzhuo/ycb-tools).
    ```bash
    cd $PointNetGPD_FOLDER/data
    git clone https://github.com/lianghongzhuo/ycb-tools
    cd ycb-tools
    python download_ycb_dataset.py rgbd_512
    ```
2. Manage your dataset at: `$PointNetGPD_FOLDER/PointNetGPD/data`
    Every object should have a folder, structure like this:
    ```
    ├002_master_chef_can
    |└── google_512k
    |    ├── nontextured.obj (generated by pcl-tools)
    |    ├── nontextured.ply
    |    ├── nontextured.sdf (generated by SDFGen)
    |└── rgbd
    |    ├── *.jpg
    |    ├── *.h5
    |    ├── ...
    ├003_cracker_box
    └004_sugar_box
    ...
    ```
3. Install SDFGen from [GitHub](https://github.com/jeffmahler/SDFGen.git):
    ```bash
    cd $PointNetGPD_FOLDER
    git clone https://github.com/jeffmahler/SDFGen.git
    cd SDFGen && mkdir build && cd build && cmake .. && make
    ```
4. Install [Open3D](http://www.open3d.org/docs/latest/getting_started.html)
    ```bash
    pip install open3d
    ```
5. Generate `nontextured.sdf` file and `nontextured.obj` file using `pcl-tools` and `SDFGen` by running:
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python read_file_sdf.py
    ```
6. Generate dataset by running the code:
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python generate-dataset-canny.py [prefix]
    ```
    where `[prefix]` is optional, it will add a prefix on the generated files.

## Visualization tools
- Visualization grasps
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python read_grasps_from_file.py
    ```
    Note:
    - This file will visualize the grasps in `$PointNetGPD_FOLDER/PointNetGPD/data/ycb_grasp/` folder

- Visualization object normal
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python Cal_norm.py
    ```
This code will check the norm calculated by `meshpy` and `pcl` library.

## Training the network
1. Data prepare:
    ```bash
    cd $PointNetGPD_FOLDER/PointNetGPD/data
    ```

    Make sure you have the following files, The links to the dataset directory should add by yourself:
    ```
    ├── google2cloud.csv  (Transform from google_ycb model to ycb_rgbd model)
    ├── google2cloud.pkl  (Transform from google_ycb model to ycb_rgbd model)
    └── ycb_grasp  (generated grasps)
    ```

    Generate point cloud from RGB-D image, you may change the number of process running in parallel if you use a shared host with others
    ```bash
    cd $PointNetGPD_FOLDER/PointNetGPD
    python ycb_cloud_generate.py
    ```
    Note: Estimated running time at our `Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz` dual CPU with 56 Threads is 36 hours. Please also remove objects beyond the capacity of the gripper.

1. Run the experiments:
    ```bash
    cd $PointNetGPD_FOLDER/PointNetGPD
    ```

    Launch a tensorboard for monitoring
    ```bash
    tensorboard --log-dir ./assets/log --port 8080
    ```

    and run an experiment for 200 epoch
    ```
    python main_1v.py --epoch 200 --mode train --batch-size x (x>=16)
    ```

    File name and corresponding experiment:
    ```
    main_1v.py        --- 1-viewed point cloud, 2 class
    main_1v_mc.py     --- 1-viewed point cloud, 3 class
    main_1v_gpd.py    --- 1-viewed point cloud, GPD
    main_fullv.py     --- Full point cloud, 2 class
    main_fullv_mc.py  --- Full point cloud, 3 class
    main_fullv_gpd.py --- Full point cloud, GPD
    ```

    For GPD experiments, you may change the input channel number by modifying `input_chann` in the experiment scripts(only 3 and 12 channels are available)

## Using the trained network

1. Get UR5 robot state:

    Goal of this step is to publish a ROS parameter tell the environment whether the UR5 robot is at home position or not.
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python get_ur5_robot_state.py
    ```
2. Run perception code:
    This code will take depth camera ROS info as input, and gives a set of good grasp candidates as output.
    All the input, output messages are using ROS messages.
    ```bash
    cd $PointNetGPD_FOLDER/dex-net/apps
    python kinect2grasp.py
    ```

    ```
    arguments:
    -h, --help                 show this help message and exit
    --cuda                     using cuda for get the network result
    --gpu GPU                  set GPU number
    --load-model LOAD_MODEL    set witch model you want to use (rewrite by model_type, do not use this arg)
    --show_final_grasp         show final grasp using mayavi, only for debug, not working on multi processing
    --tray_grasp               not finished grasp type
    --using_mp                 using multi processing to sample grasps
    --model_type MODEL_TYPE    selet a model type from 3 existing models
    ```

## Citation
If you found PointNetGPD useful in your research, please consider citing:

```plain
@inproceedings{liang2019pointnetgpd,
  title={{PointNetGPD}: Detecting Grasp Configurations from Point Sets},
  author={Liang, Hongzhuo and Ma, Xiaojian and Li, Shuang and G{\"o}rner, Michael and Tang, Song and Fang, Bin and Sun, Fuchun and Zhang, Jianwei},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019}
}
```

## Acknowledgement
- [gpg](https://github.com/atenpas/gpg)
- [gpd](https://github.com/atenpas/gpd)
- [dex-net](https://github.com/BerkeleyAutomation/dex-net)
- [meshpy](https://github.com/BerkeleyAutomation/meshpy)
- [SDFGen](https://github.com/christopherbatty/SDFGen)
- [pyntcloud](https://github.com/daavoo/pyntcloud)
- [metu-ros-pkg](https://github.com/kadiru/metu-ros-pkg)
- [mayavi](https://github.com/enthought/mayavi)
