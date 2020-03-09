# VNwithSMap

presentation is [here](https://github.com/es6rc/VNwithSemanticMap/blob/master/Demo/Pre.pdf)

Visual Navigation in indoor environment with Top-down Semantic Map. 

by taking advantage of [value iteration network](https://github.com/avivt/VIN), the action policy network is based on the value map generated.

### Semantic Map Visualization
Each Semantic Map has a size of `H X W X C` where `H` and `W` stands for height and width, and `C` is the number of object categories. Each cell on Semantic Map has a vector of object occupancies.

![image](https://github.com/es6rc/VNwithSemanticMap/blob/master/Demo/SMap.png)

### Robot Pose Visualiaztion
![image](https://github.com/es6rc/VNwithSemanticMap/blob/master/Demo/visualization.png)

### Network
![image](https://github.com/es6rc/VNwithSemanticMap/blob/master/Demo/Network.png)

## Preprocess

### Add patches to House3D dataset

* Add functions in `preprocess/patch_core.py` to the `House3D/House3D/core.py` file under the `Environment` class.

* Add functions in `preprocess/patch_house.py` to the `House3D/House3D/house.py` file under the `House` class.

* Integrate the `colormap.csv` to the `House` class by adding line **7** to **13** to `House` Class. This also generates `self.smap` and `self.smap_img` to the Class.

### Parsing necessary data

* `preprocess/genhouseinfo.py` generates the necessary house information for local semantic map as well. Modify the `HOUSEDIR`, `CONFIGFILEPATH` and `house_ids`. run by ``python preprocess/genhouseinfo.py``.

* `preprocess/gensmap.py` provides with a `gensmap` class and generates the local semantic map of **94** classes at every location given in a **`map.txt` file previously generated (not included in the repo)**. change the `house_ids` and `lmapszs` to desired value and run by ``python preprocess/gensmap.py``. 
* `get_tar_star_minsteps_aseq.py` generates **action sequency** for every location in given **`map.txt` file** for each given target.

## Postprocess

### Visualize the trajectory and learnt reward map and value map
* `res_vis.py` gives a solution to generate all frames of learnt reward map and value map and local semantic map along the trajctory.


## Training & Testing

### Interactor
* `src/navi_env.py` interacts with the aforementioned generated data.

* `src/multienv.py` builds upon `nav_env.py` and interacts with multiple environments and enable multiple agents approaching different target (one agent one target).

### Train & Test
* `tf_code/nav_agent_release.py` initialize training or testing. 

