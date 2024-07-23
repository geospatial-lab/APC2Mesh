# APC2Mesh

__*APC2Mesh*__ is the repo for the implementation of our paper "*[APC2Mesh: Bridging the gap from occluded building façades to full 3D models](https://www.sciencedirect.com/science/article/pii/S0924271624001692)*".

> [!NOTE]
This [Building3D benchmark dataset](https://building3d.ucalgary.ca/reconstruction.php) is still undergoing curation and revisions. For now, the website only has data for Tallinn City. [Here](https://drive.google.com/drive/my-drive) is a subset of the Building3D dataset used in APC2Mesh. The `sample-data` folder has 3 sub-folders `mesh`, `xyz`, and `wframe` which represent the groundtruth meshes, partial point sets, and wireframes of buildings from different cities.

``` bash

APC2Mesh/
│
├── README.md # Project documentation
├── sdf_try.py # The python script for pre-processing the files in `sample-data`
├── src/ # Source code
│ ├── main.py
│ ├── module1.py
│ └── module2.py
├── tests/ # Test files
│ ├── test_main.py
│ └── test_module1.py
├── data/ # Data files
│ ├── raw/
│ │ └── dataset.csv
│ └── processed/
│ └── cleaned_data.csv
└── assets/ # Image and other assets
├── logo.png
└── screenshot.png

```

###Data Preprocessing
