# APC2Mesh

__*APC2Mesh*__ is the repo for the implementation of our paper "*[APC2Mesh: Bridging the gap from occluded building façades to full 3D models](https://www.sciencedirect.com/science/article/pii/S0924271624001692)*".

> [!NOTE]
This [Building3D benchmark dataset](https://building3d.ucalgary.ca/reconstruction.php) is still undergoing curation and revisions. For now, the website only has data for Tallinn City. [Here](https://drive.google.com/drive/my-drive) is a subset of the Building3D dataset used in APC2Mesh. The `sample-data` folder has 3 sub-folders `mesh`, `xyz`, and `wframe` which represent the groundtruth meshes, partial point sets, and wireframes of buildings from different cities.


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

I
