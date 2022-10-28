
<p align="center"> <b> AI Tools for the Modern Problem Solver
</b> 


<p align="center">
<img src="robot.png" width="68">
</p>


A curated list of AI tools and techniques to solve problems. No background knowledge assumed.
 
Here AI is an umbrella term for anything that has to do with learning and/or data, including techniques from Statistics, Data Science, Machine Learning, Artificial General Intelligence and related fields.
 
The goal is to let you know that these tools exist, giving a basic usage example, resources and weaknesses.

Contributions to the list are welcome.



-----------

**Template**
 
*Superpower*: What it does.

*How hard*: How hard to use it without understanding the theory behind (debugging, parameter selection, etc.)

*Libraries*: Suggested implementations.

*Example*: Basic tutorial. 

*Weak Points*: When it tends to not work well.

*Other*: Other!


-------------
**Table of Contents**

- [Neural Network Classifier (Supervised Learning, Deep Learning)](#neural-network-classifier-supervised-learning-deep-learning)
- [Neural Network Object Detector (Supervised Learning, Deep Learning)](#neural-network-object-detector-supervised-learning-deep-learning)
- [Semantic, Instance and Panoptic Segmentation (Supervised Learning, Deep Learning)](#semantic-instance-and-panoptic-segmentation-supervised-learning-deep-learning)
- [Transfer Learning (Deep Learning)](#transfer-learning-deep-learning)
- [Decision Trees (Supervised Learning)](#decision-trees-supervised-learning)
- [Random Forest (Supervised Learning)](#random-forest-supervised-learning)
- [AutoML (Supervised or Unsupervised Learning)](#automl-supervised-or-unsupervised-learning)
- [Clustering (Unsupervised Learning)](#clustering-unsupervised-learning)
- [Reinforcement Learning](#reinforcement-learning)
- [Artificial General Intelligence](#artificial-general-intelligence)

-------------


### Neural Network Classifier (Supervised Learning, Deep Learning)
*Superpower*: Classify objects in images, text, video and more.

*How hard*: Medium. Popular defaults are good enough to get far, the bottleneck is getting good data. Debugging is tricky.  

*Libraries*: Detectron2.

*Example*: [Vision classifier on CIFAR10 dataset](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 

*Weak Points*: Needs a lot of labelled data to be highly accurate, hard to interpret, hard to quantify prediction confidence, long training time. 

*Other*: Top choice to classify high level concepts such as "cat vs dog" given an image. 

---

### Neural Network Object Detector (Supervised Learning, Deep Learning)
*Superpower*: Detect objects in images and videos, usually returning a bounding box.

*How hard*: Same as Neural Network Classifier.   

*Libraries*: Detectron2, yolov5.

*Example*: [Object Detection with Detectron2](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch/) 

*Weak Points*: Same as Neural Network Classifier.  

*Other*: Production ready tool, working in real time. 

---

### Semantic, Instance and Panoptic Segmentation (Supervised Learning, Deep Learning)
*Superpower*: Detect instances of objects inside an image, pixel per pixel. Semantic segmentation treats them as aggregates detections, while Instance segmentation singles out individual objects but do not consider every single pixel. Panoptic segmentation combines both. 

*How hard*: Same as Neural Network Classifier.   

*Libraries*: Detectron2.

*Example*: [Instance Segmentation with Detectron2](https://gilberttanner.com/blog/detectron2-train-a-instance-segmentation-model/) 

*Weak Points*: Data labelling is very time consuming.  

*Other*:   

---

### Transfer Learning (Deep Learning)
*Superpower*: Train a neural network with little data, by reusing a pre-trained network as a starting point.

*How hard*: Easy.

*Libraries*: Your favorite DL library would do.

*Example*: [Transfer Learning For Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 

*Weak Points*: May inherit some restrictions from the original model.

*Other*: New predicted classes should be somehow similar to the pre-trained ones for better results. Otherwise just train longer.


---

### Decision Trees (Supervised Learning)
*Superpower*: Classification and regression with little data preparation and no (or few) parameters.

*How hard*: Easy.

*Libraries*: Sklearn.

*Example*: [Decision Trees with Sklearn](https://scikit-learn.org/stable/modules/tree.html) 

*Weak Points*: Tend to overfit or be unstable, not suited for high-dim data (e.g. images). 

*Other*: Great explainability (can be mapped to yes/no questions), great for tabular data.

---

### Random Forest (Supervised Learning)
*Superpower*: Classification and regression with little data preparation. 

*How hard*: Easy.

*Libraries*: Sklearn.

*Example*: [Random Forest with Sklearn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py) 

*Weak Points*: Less explainable than decision trees, not suited for high-dim data (e.g. images). 

*Other*: Less overfit than decision trees, great for tabular data.

---

### AutoML (Supervised or Unsupervised Learning)
*Superpower*: Automate the algorithm selection, remove human bias.

*How hard*: Easy.

*Libraries*: [auto-sklearn](https://github.com/automl/auto-sklearn)
 

*Example*: [Minimal AutoML tutorial](https://automl.github.io/auto-sklearn/master/#example) 

*Weak Points*: Very slow, lack of control.

*Other*: Can make sense for the initial exploration of problems with no clear angle of attack.

---

### Clustering (Unsupervised Learning)
*Superpower*: Unsupervised clustering of data into classes.

*How hard*: Easy.

*Libraries*: [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html#n)
 
*Example*: [K-Means clustering on the handwritten digits data](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py) 

*Weak Points*: Not as good as deep learning for complex problems.

*Other*: Many flavours of clustering are available. 

---

### Reinforcement Learning 
*Superpower*: Learning how to act optimally in a dynamic environment in order to maximize a reward.

*How hard*: Hard. Hard debugging. It may be hard to create a meaningful reward function. Even top algorithms can be unstable.

*Libraries*: [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/). [RLib](https://docs.ray.io/en/latest/rllib/index.html) is more production-ready, but also less user friendly. Check also the [Gym Environments docs](https://www.gymlibrary.dev/).
 
*Example*: [Cart Pole Environment in Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) 

*Weak Points*: Slow to train, needs a lot of data. Requires a good simulator of the environment.

*Other*: Applicable to vastly different problems, from [nuclear fusion](https://www.deepmind.com/blog/accelerating-fusion-science-through-learned-plasma-control) to [pure maths](https://www.nature.com/articles/s41586-022-05172-4). State of the art techniques are usually deep learning based (Deep Reinforcement Learning).

---
### Artificial General Intelligence 
*Superpower*: Solving pretty much any task, given enough time and resources.

*How hard*: Easy?

*Libraries*: ...
 
*Example*: ...

*Weak Points*: Hard to build one.

*Other*: It may be conscious! 






-----------

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

<small><i><a href="https://www.flaticon.com/free-icons/robot" title="robot icons">Robot icons created by Freepik - Flaticon</a></i></small>
