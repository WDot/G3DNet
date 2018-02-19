# G3DNet
General-Purpose Point Cloud Feature Extractor

This is code and a pretrained network for extracting features from point cloud objects. If your 3D objects can be sampled as roughly 500 points in a point cloud, this should generate a reasonable nonlinear feature representation for classification or other applications. The code is sufficient to preprocess 3D mesh data and train the network. The network can take a long time to train though (in our experience, at least a month on a single Titan X GPU).

The pretrained network can be downloaded from https://drive.google.com/file/d/1VQ2nfBZfeWv60uQzbLv1yHrHTGbt-1Kk/view?usp=sharing and unzipped into the root of code under a directory named /snapshots. The run.sh script in the root directory has options to run the three separate training steps as well as options to extract features from the two pretrained models, G3DNet18 and G3DNet26. Read the contents of run.sh and make sure to fill in information as necessary (such as what dataset you're training on and its root directory).

If you have issues with this code, please register an issue on this Github page. If you find it useful, please cite us:
```
  @InProceedings{Long2015,
  author = {Dominguez, Miguel and Dhamdhere, Rohan and Petkar, Atir and Jain, Saloni and  Sah, Shagan and Ptucha, Ray},
  title = {General-Purpose Point Cloud Feature Extractor},
  booktitle = {Winter Applications of Computer Vision 2018},
  month = {March},
  year = {2018}
  }
```
