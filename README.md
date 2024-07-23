# APC2Mesh

__*APC2Mesh*__ is the repo for the implementation of our paper "*[APC2Mesh: Bridging the gap from occluded building façades to full 3D models](https://www.sciencedirect.com/science/article/pii/S0924271624001692)*".

> [!NOTE]
This [Building3D benchmark dataset](https://building3d.ucalgary.ca/reconstruction.php) is still undergoing curation and revisions. For now, the website only has data for Tallinn City. [Here](https://drive.google.com/file/d/17Wdi3ceJxMuyhHVmBjMeEyHod04p6QoQ/view?usp=sharing) is a subset of the Building3D dataset used in APC2Mesh. The `sample-data` folder has 3 sub-folders `mesh`, `xyz`, and `wframe` which represent the groundtruth meshes, partial point sets, and wireframes of buildings from different cities.


### Important project files to highlight
``` bash

APC2Mesh/
│
├── README.md # Project documentation
├── sdf_try.py # The python script for pre-processing the files in `sample-data`
├── dataset_pcc.py # The python script for the custom dataloader for point completion task
├── .vscode/ # Source code
│ └── tasks.json
├── ablations/ # Test files
│ └── pcc.py # script used to train point completion model
├── main.py # script used to run the reconstruction phase of the project
└── Dockerfile20 # Used in conjunction with `.vscode\tasks.json` to run project in Docker

```

### Prerequisites
The necessary python packages and environmental settings we used in building this project can be found in `Dockerfile20`.
To run this repo, either:
  1. Use the packages and environmental specifications from the dockerfile to create your own local workspace OR
  2. build the Dockerfile and run the project in a Docker container.

### Citation
If you use APC2Mesh in a scientific work, please consider citing the paper:

``` bibtex
@article{akwensi2024apc2mesh,
  title = {APC2Mesh: Bridging the gap from occluded building façades to full 3D models},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {211},
  pages = {438-451},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.04.009},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271624001692},
  author = {Perpetual Hope Akwensi and Akshay Bharadwaj and Ruisheng Wang}
}
```
