<h2 align="center">
  <b>Trace3D: Consistent Segmentation Lifting via</b>
  <br>
  <b>Gaussian Instance Tracing</b>

  <b><i>ICCV 2025 </i></b>
</h2>

<p align="center">
    <a href="https://github.com/shy2000">Hongyu Shen </a><sup>1,2,*</sup>
    <a href="https://dali-jack.github.io/Junfeng-Ni/">Junfeng Ni</a><sup>2,3,*</sup>,
    <a href="https://yixchen.github.io/">Yixin Chen<sup>✉</sup></a><sup>2</sup>,
    <a href="">Weishuo Li</a><sup>2</sup>,
    <a href="https://peimingtao.github.io/">Mingtao Pei</a><sup>1</sup>,
    <a href="https://siyuanhuang.com/">Siyuan Huang<sup>✉</sup></a><sup>2</sup>
    <br>
    <a style="font-size: 0.9em; padding: 0.5em 0;"><sup>✉</sup> indicates corresponding author</a> &nbsp&nbsp 
  <a style="font-size: 0.9em; padding: 0.5em 0;"><sup>*</sup> these authors contributed equally to this work</a> &nbsp&nbsp 
    <sup>1</sup>Beijing Institute of Technology
    <br>
    <sup>2</sup>State Key Laboratory of General Artificial Intelligence, BIGAI &nbsp&nbsp 
    <sup>3</sup>Tsinghua University
</p>

<p align="center">
    <a href='https://arxiv.org/pdf/2508.03227'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper arXiv'>
    </a>
    <!-- <a href=''>
      <img src='https://img.shields.io/badge/Video-green?style=plastic&logo=arXiv&logoColor=green' alt='Video Demo'>
    </a> -->
    <a href='https://trace-3d.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

<p align="center">
    <img src="assets/teaser.jpg" width=90%>
</p>

Trace3D leverages the proposed Gaussian Instance Tracing to enhance multi-view consistency and reduce ambiguous Gaussians, resulting in high-quality 3D instance segmentation.

## Installation
- Tested System: Ubuntu 22.04, CUDA 11.8
- Tested GPUs: RTX4090

1. Basic environment
```bash
conda create -n trace3d python=3.10
conda activate trace3d
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
2. SAM for segmentation
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
## Data
```bash
    data
    ├── nerf_llff_data  # Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │       ├── [sparse/0] (colmap results)
    │       └── [images|images_2|images_4|images_8]
    │
    └── replica		# Link: https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0
        └── [office_0|room_0|...]
            ├── traj_w_c.txt
            ├── [sparse/0] (colmap results)
            └── [rgb|depth|sam|]
```
## Training
Get SAM masks
```bash
python get_sam_masks.py --sam_checkpoint {SAM_CKPT_PATH} --file_path {IMAGE_FOLDER}
```
Before running: please specify the information in the scripts (e.g. `replica.sh`). More options can be found in `conf/` and `arguments/` and you can them adjusted in config file.
```bash
#--- Edit the config file replica.sh
dataset=replica_900
path=./data/${dataset}
scene='room_0' 
```
Scene reconstruction
```bash
bash replica.sh train_rgb
```

Merge patch masks 
```bash
bash replica.sh merge_patches
```
Delete Ambiguous Gaussians
```bash
bash replica.sh remove_ab_gaus
```
Contrastive lifting
```bash
bash replica.sh train_contra
```
<!-- ### 2.Merge patch masks  

### 3.Delete Ambiguous Gaussians -->


## Evaluation
3D Object Extraction
```bash
bash replica.sh eval_3d
```
如果想把每个物体对应的高斯子集导出为 `.ply`（可直接用 MeshLab / CloudCompare / Open3D 等 3D viewer 打开），先在 `eval_3d` 时保存 mask，再执行导出脚本：
```bash
# 1) 先生成每个物体的高斯 mask（默认保存在 ${model_path}/objects/*.pt）
python evaluation/eval_3d.py \
  -s {SOURCE_PATH} \
  -m {MODEL_PATH} \
  --save_path {VIS_SAVE_PATH} \
  --result_save_path {RESULT_SAVE_DIR} \
  --method split \
  --start_checkpoint {CHECKPOINT_PATH} \
  --save_gaus_mask

# 2) 导出每个物体的 gaussian 子集为 ply（默认输出到 ${model_path}/objects_ply）
python evaluation/export_gaus_ply.py \
  -s {SOURCE_PATH} \
  -m {MODEL_PATH} \
  --start_checkpoint {CHECKPOINT_PATH}
```
如果你需要导出**整个场景**（而不是对象子集）的 `.ply`，可直接从 checkpoint 导出：
```bash
python evaluation/export_scene_ply.py \
  -s {SOURCE_PATH} \
  -m {MODEL_PATH} \
  --start_checkpoint {CHECKPOINT_PATH} \
  --save_path {MODEL_PATH}/scene.ply \
  --unity_compatible
```
说明：
- 默认不加 `--unity_compatible` 时，保持 Trace3D 原始导出语义；
- 加 `--unity_compatible` 时，会导出更适合 Unity Gaussian Loader 的参数（激活后的 opacity/scale、归一化 rotation、并补 `scale_2=1`）。
可选参数：
- `--gaus_mask_path`：指定单个 `*_gaus_mask.pt` 文件，或包含多个 mask 的目录（默认 `${model_path}/objects`）。
- `--save_path`：指定导出 ply 的目录（默认 `${model_path}/objects_ply`）。

如果要做对象级编辑（例如在 Unity 中拖动每个对象的包围盒），可以在同一批 `*_gaus_mask.pt` 上导出对象包围盒：
```bash
# 3) 导出每个对象的包围盒（AABB + OBB）到 JSON
python evaluation/export_object_bboxes.py \
  -s {SOURCE_PATH} \
  -m {MODEL_PATH} \
  --start_checkpoint {CHECKPOINT_PATH} \
  --gaus_mask_path {MODEL_PATH}/objects \
  --save_path {MODEL_PATH}/objects_bbox.json
```

导出的 `objects_bbox.json` 同时包含：
- 原始右手系下的 `aabb` / `obb`；
- 以及用于 Unity 的左手系 `unity.aabb` / `unity.obb`（已做 z 翻转）。

Unity 侧最小接入方式：
- 将 `unity/Trace3DObjectEdit/Trace3DBboxLoader.cs` 与 `unity/Trace3DObjectEdit/Trace3DBoxDrag.cs` 放入 Unity 工程；
- 把 `objects_bbox.json` 放到 Unity `Assets` 下并作为 `TextAsset` 引用给 `Trace3DBboxLoader.bboxJson`；
- 在场景中挂载 `Trace3DBboxLoader`，点击 Inspector 的 `Load Boxes`（或运行时调用）即可生成可拖动包围盒。

如果要直接把导出的 `.ply` 渲染成“模型风格”（体素网格，而不是点云）：
- 将 `unity/Trace3DObjectEdit/Trace3DPlyVoxelMeshRenderer.cs` 放入 Unity 工程；
- 在场景中创建空物体并挂载该脚本；
- 通过 `plyAsset`（TextAsset）或 `plyFilePath`（绝对路径）指定 PLY；
- 点击 Inspector 的 `Build Mesh From PLY`；
- 该脚本同时支持 `ascii`、`binary_little_endian`、`binary_big_endian` 三种 PLY 格式。

Novel View 2D Instance Segmentation
```bash
bash replica.sh eval         
```

## Acknowledgements
Some codes are borrowed from  [Egolifter](https://github.com/facebookresearch/egolifter), [SA3D](https://github.com/Jumpat/SegmentAnythingin3D), [Omniseg3D](https://github.com/THU-luvision/OmniSeg3D), [FlashSplat](https://github.com/florinshen/FlashSplat) and [Gaussian-Editor](https://github.com/buaacyw/GaussianEditor). We thank all the authors for their great work. 

## Citation

```bibtex
@inproceedings{shen2025trace3d,
  title={Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing},
  author={Shen, Hongyu and Ni, Junfeng, and Chen, Yixin and Li, Weishuo and Pei, Mingtao and Huang, Siyuan},
  booktitle=ICCV,
  year={2025}
}
```
