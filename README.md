# Awesome - Most Cited Deep Learning Papers 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of the most cited deep learning papers (since 2010)

I believe that there exist *classic* deep learning papers which are worth reading regardless of their applications. Rather than providing overwhelming amount of papers, I would like to provide a *curated list* of the classic deep learning papers which can be considered as *must-reads* in some area. 

## Awesome list criteria

- **2016** :  Based on discussions
- **2015** :  +100 citations (:sparkles: +200)
- **2014** :  +200 citations (:sparkles: +400)
- **2013** :  +300 citations (:sparkles: +600)
- **2012** :  +400 citations (:sparkles: +800)
- **2011** :  +500 citations (:sparkles: +1000)
- **2010** :  +600 citations (:sparkles: +1200)

*I need your [contributions](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md)!*

## Table of Contents 

* [Survey / Review](#survey--review)
* [Theory / Future](#theory--future)
* [Optimization / Regularization](#optimization--regularization)
* [Network Models](#network-models)
* [Image](#image)
* [Caption](#caption)
* [Video / Human Activity](#video--human-activity)
* [Word Embedding](#word-embedding)
* [Machine Translation / QnA](#machine-translation--qna)
* [Speech / Etc.](#speech--etc)
* [RL / Robotics](#rl--robotics)
* [Unsupervised](#unsupervised)
* [Hardware / Software](#hardware--software)

(Total 84 papers)

### Survey / Review
- Deep learning (Book, 2016), Goodfellow et al. *(Bengio)* [[html]](http://www.deeplearningbook.org/) 
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[html]](http://www.nature.com/nature/journal/v521/n7553/abs/nature14539.html) :sparkles:
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](http://arxiv.org/pdf/1404.7828) :sparkles:
- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](http://arxiv.org/pdf/1206.5538) :sparkles:

### Theory / Future
- Distilling the knowledge in a neural network (2015), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1503.02531)
- Deep neural networks are easily fooled: High confidence predictions for unrecognizable images (2015), A. Nguyen et al. [[pdf]](http://arxiv.org/pdf/1412.1897)
- How transferable are features in deep neural networks? (2014), J. Yosinski et al. *(Bengio)* [[pdf]](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- Why does unsupervised pre-training help deep learning (2010), E. Erhan et al. *(Bengio)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)

### Optimization / Regularization
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Loffe and C. Szegedy [[pdf]](http://arxiv.org/pdf/1502.03167) :sparkles:
- Delving deep into rectifiers: Surpassing human-level performance on imagenet classification (2015), K. He et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) :sparkles:
- Dropout: A simple way to prevent neural networks from overfitting (2014), N. Srivastava et al. *(Hinton)* [[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) :sparkles:
- Adam: A method for stochastic optimization (2014), D. Kingma and J. Ba [[pdf]](http://arxiv.org/pdf/1412.6980)
- Spatial pyramid pooling in deep convolutional networks for visual recognition (2014), K. He et al. [[pdf]](http://arxiv.org/pdf/1406.4729)
- Regularization of neural networks using dropconnect (2013), L. Wan et al. *(LeCun)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf)
- Improving neural networks by preventing co-adaptation of feature detectors (2012), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1207.0580.pdf) :sparkles:
- Random search for hyper-parameter optimization (2012) J. Bergstra and Y. Bengio [[pdf]](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a)

### Network Models
- Deep residual learning for image recognition (2016), K. He et al. *(Microsoft)* [[pdf]](http://arxiv.org/pdf/1512.03385) :sparkles:
- Going deeper with convolutions (2015), C. Szegedy et al. *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) :sparkles:
- Fast R-CNN (2015), R. Girshick [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) :sparkles:
- Fully convolutional networks for semantic segmentation (2015), J. Long et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) :sparkles:
- Very deep convolutional networks for large-scale image recognition (2014), K. Simonyan and A. Zisserman [[pdf]](http://arxiv.org/pdf/1409.1556) :sparkles:
- OverFeat: Integrated recognition, localization and detection using convolutional networks (2014), P. Sermanet et al. *(LeCun)* [[pdf]](http://arxiv.org/pdf/1312.6229)
- Visualizing and understanding convolutional networks (2014), M. Zeiler and R. Fergus [[pdf]](http://arxiv.org/pdf/1311.2901) :sparkles:
- Maxout networks (2013), I. Goodfellow et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1302.4389v4)
- ImageNet classification with deep convolutional neural networks (2012), A. Krizhevsky et al. *(Hinton)* [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) :sparkles:
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) :sparkles:
- Deep sparse rectifier neural networks (2011), X. Glorot et al. *(Bengio)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)

### Image
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](http://arxiv.org/pdf/1409.0575) :sparkles:
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015), S. Ren et al. [[pdf]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  :sparkles:
- DRAW: A recurrent neural network for image generation (2015), K. Gregor et al. [[pdf]](http://arxiv.org/pdf/1502.04623)
- Rich feature hierarchies for accurate object detection and semantic segmentation (2014), R. Girshick et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Learning and transferring mid-Level image representations using convolutional neural networks (2014), M. Oquab et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification (2014), Y. Taigman et al. *(Facebook)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) :sparkles:
- Decaf: A deep convolutional activation feature for generic visual recognition (2013), J. Donahue et al. [[pdf]](http://arxiv.org/pdf/1310.1531) :sparkles:
- Learning Hierarchical Features for Scene Labeling (2013), C. Farabet et al. *(LeCun)* [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)
- Learning mid-level features for recognition (2010), Y. Boureau *(LeCun)* [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)

### Caption
- Show, attend and tell: Neural image caption generation with visual attention (2015), K. Xu et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1502.03044) :sparkles:
- Show and tell: A neural image caption generator (2015), O. Vinyals et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) :sparkles:
- Long-term recurrent convolutional networks for visual recognition and description (2015), J. Donahue et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Deep visual-semantic alignments for generating image descriptions (2015), A. Karpathy and L. Fei-Fei [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.html) :sparkles:


### Video / Human Activity
- Large-scale video classification with convolutional neural networks (2014), A. Karpathy et al. *(FeiFei)* [[pdf]](vision.stanford.edu/pdf/karpathy14.pdf) :sparkles:
- DeepPose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
- A survey on human activity recognition using wearable sensors (2013), O. Lara and M. Labrador [[pdf]](http://romisatriawahono.net/lecture/rm/survey/computer%20vision/Lara%20-%20Human%20Activity%20Recognition%20-%202013.pdf)
- 3D convolutional neural networks for human action recognition (2013), S. Ji et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_JiXYY10.pdf)
- Action recognition with improved trajectories (2013), H. Wang and C. Schmid [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf)
- Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis (2011), Q. Le et al. [[pdf]](http://robotics.stanford.edu/~wzou/cvpr_LeZouYeungNg11.pdf)

### Word Embedding
- Glove: Global vectors for word representation (2014), J. Pennington et al. [[pdf]](http://llcao.net/cu-deeplearning15/presentation/nn-pres.pdf) :sparkles:
- Distributed representations of sentences and documents (2014), Q. Le and T. Mikolov [[pdf]](http://arxiv.org/pdf/1405.4053) *(Google)* :sparkles:
- Distributed representations of words and phrases and their compositionality (2013), T. Mikolov et al. *(Google)* [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) :sparkles:
- Efficient estimation of word representations in vector space (2013), T. Mikolov et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1301.3781) :sparkles:
- Word representations: a simple and general method for semi-supervised learning (2010), J. Turian *(Bengio)* [[pdf]](http://www.anthology.aclweb.org/P/P10/P10-1040.pdf)

### Machine Translation / QnA
- Towards ai-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](http://arxiv.org/pdf/1502.05698)
- Recurrent Continuous Translation Models (2013), N. Kalchbrenner and P. Blunsom. [[pdf]](http://www.aclweb.org/anthology/D13-1176)
- Neural machine translation by jointly learning to align and translate (2014), D. Bahdanau et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1409.0473) :sparkles:
- Sequence to sequence learning with neural networks (2014), I. Sutskever et al. [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- Learning phrase representations using RNN encoder-decoder for statistical machine translation (2014), K. Cho et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1406.1078)
- A convolutional neural network for modelling sentences (2014), N. kalchbrenner et al. [[pdf]](http://arxiv.org/pdf/1404.2188v1)
- Convolutional neural networks for sentence classification (2014), Y. Kim [[pdf]](http://arxiv.org/pdf/1408.5882)
- The stanford coreNLP natural language processing toolkit (2014), C. Manning et al. [[pdf]](http://www.surdeanu.info/mihai/papers/acl2014-corenlp.pdf) :sparkles:
- Recursive deep models for semantic compositionality over a sentiment treebank (2013), R. Socher et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf) :sparkles:
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](http://arxiv.org/pdf/1103.0398) :sparkles:
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)

### Speech / Etc.
- Speech recognition with deep recurrent neural networks (2013), A. Graves *(Hinton)* [[pdf]](http://arxiv.org/pdf/1303.5778.pdf)
- Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups (2012), G. Hinton et al. [[pdf]](http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf) :sparkles:
- Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition (2012) G. Dahl et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf) :sparkles:
- Acoustic modeling using deep belief networks (2012), A. Mohamed et al. *(Hinton)* [[pdf]](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)


### RL / Robotics
- Mastering the game of Go with deep neural networks and tree search, D. Silver et al. *(DeepMind)* [[pdf]](Mastering the game of Go with deep neural networks and tree search)
- Human-level control through deep reinforcement learning (2015), V. Mnih et al. *(DeepMind)* [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf) :sparkles:
- Deep learning for detecting robotic grasps (2015), I. Lenz et al. [[pdf]](http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
- Playing atari with deep reinforcement learning (2013), V. Mnih et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1312.5602.pdf))

### Unsupervised
- Generative adversarial nets (2014), I. Goodfellow et al. *(Bengio)* [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- Auto-Encoding Variational Bayes (2013), D. Kingma and M. Welling [[pdf]](http://arxiv.org/pdf/1312.6114)
- Building high-level features using large scale unsupervised learning (2013), Q. Le et al. [[pdf]](http://arxiv.org/pdf/1112.6209) :sparkles:
- Contractive auto-encoders: Explicit invariance during feature extraction (2011), S. Rifai et al. *(Bengio)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf)
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- Pixel Recurrent Neural Networks (2016), van den Oord, et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)

### Hardware / Software
- TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems (2016), M. Abadi et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1603.04467)
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](http://arxiv.org/pdf/1408.5093) :sparkles:
- Theano: new features and speed improvements (2012), F. Bastien et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1211.5590)

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Terry T. Um](https://www.facebook.com/terryum.io/) has waived all copyright and related or neighboring rights to this work.
