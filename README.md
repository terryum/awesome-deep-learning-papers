# Awesome - Most Cited Deep Learning Papers 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of the most cited deep learning papers (since 2010)

I believe that there exist *classic* deep learning papers which are worth reading regardless of their applications. Rather than providing overwhelming amount of papers, I would like to provide a *curated list* of the classic deep learning papers which can be considered as *must-reads* in some area. 

## Awesome list criteria

- **2016** :  +30 citations (:sparkles: +50)
- **2015** :  +100 citations (:sparkles: +200)
- **2014** :  +200 citations (:sparkles: +400)
- **2013** :  +300 citations (:sparkles: +600)
- **2012** :  +400 citations (:sparkles: +800)
- **2011** :  +500 citations (:sparkles: +1000)
- **2010** :  +600 citations (:sparkles: +1200)

*I need your contributions! Please read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) before you make a pull request.*

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
* [Papers Worth Reading](#papers-worth-reading)
* [Classic papers](#classic-papers)
* [Distinguished Researchers](#distinguished-researchers)

Total 85 papers except for the papers in *Hardware / Software*, *Papers Worth Reading*, and *Classic papers* sections.

### Survey / Review
- Deep learning (Book, 2016), Goodfellow et al. *(Bengio)* [[html]](http://www.deeplearningbook.org/) 
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[html]](http://www.nature.com/nature/journal/v521/n7553/abs/nature14539.html) :sparkles:
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](http://arxiv.org/pdf/1404.7828) :sparkles:
- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](http://arxiv.org/pdf/1206.5538) :sparkles:

### Theory / Future
- Distilling the knowledge in a neural network (2015), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1503.02531)
- Deep neural networks are easily fooled: High confidence predictions for unrecognizable images (2015), A. Nguyen et al. [[pdf]](http://arxiv.org/pdf/1412.1897)
- How transferable are features in deep neural networks? (2014), J. Yosinski et al. *(Bengio)* [[pdf]](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- Return of the devil in the details: delving deep into convolutional nets (2014), K. Chatfield et al. [[pdf]](http://arxiv.org/pdf/1405.3531) :sparkles:
- Why does unsupervised pre-training help deep learning (2010), D. Erhan et al. *(Bengio)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)

### Optimization / Regularization
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Loffe and C. Szegedy *(Google)* [[pdf]](http://arxiv.org/pdf/1502.03167) :sparkles:
- Delving deep into rectifiers: Surpassing human-level performance on imagenet classification (2015), K. He et al. *(Microsoft)* [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) :sparkles:
- Dropout: A simple way to prevent neural networks from overfitting (2014), N. Srivastava et al. *(Hinton)* [[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) :sparkles:
- Adam: A method for stochastic optimization (2014), D. Kingma and J. Ba [[pdf]](http://arxiv.org/pdf/1412.6980)
- Spatial pyramid pooling in deep convolutional networks for visual recognition (2014), K. He et al. [[pdf]](http://arxiv.org/pdf/1406.4729)
- On the importance of initialization and momentum in deep learning (2013), I. Sutskever et al. *(Hinton)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_sutskever13.pdf)
- Regularization of neural networks using dropconnect (2013), L. Wan et al. *(LeCun)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf)
- Improving neural networks by preventing co-adaptation of feature detectors (2012), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1207.0580.pdf) :sparkles:
- Random search for hyper-parameter optimization (2012) J. Bergstra and Y. Bengio [[pdf]](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a)

### Network Models
- Deep residual learning for image recognition (2016), K. He et al. *(Microsoft)* [[pdf]](http://arxiv.org/pdf/1512.03385) :sparkles:
- Region-based convolutional networks for accurate object detection and segmentation (2016), R. Girshick et al. *(Microsoft)* [[pdf]](https://www.cs.berkeley.edu/~rbg/papers/pami/rcnn_pami.pdf)
- Going deeper with convolutions (2015), C. Szegedy et al. *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) :sparkles:
- Fast R-CNN (2015), R. Girshick *(Microsoft)* [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) :sparkles:
- Fully convolutional networks for semantic segmentation (2015), J. Long et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) :sparkles:
- Very deep convolutional networks for large-scale image recognition (2014), K. Simonyan and A. Zisserman [[pdf]](http://arxiv.org/pdf/1409.1556) :sparkles:
- OverFeat: Integrated recognition, localization and detection using convolutional networks (2014), P. Sermanet et al. *(LeCun)* [[pdf]](http://arxiv.org/pdf/1312.6229)
- Visualizing and understanding convolutional networks (2014), M. Zeiler and R. Fergus [[pdf]](http://arxiv.org/pdf/1311.2901) :sparkles:
- Maxout networks (2013), I. Goodfellow et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1302.4389v4)
- Network in network (2013), M. Lin et al. [[pdf]](http://arxiv.org/pdf/1312.4400)
- ImageNet classification with deep convolutional neural networks (2012), A. Krizhevsky et al. *(Hinton)* [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) :sparkles:
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) :sparkles:
- Deep sparse rectifier neural networks (2011), X. Glorot et al. *(Bengio)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)

### Image
- Reading text in the wild with convolutional neural networks (2016), M. Jaderberg et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1412.1842)
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](http://arxiv.org/pdf/1409.0575) :sparkles:
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015), S. Ren et al. [[pdf]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  :sparkles:
- DRAW: A recurrent neural network for image generation (2015), K. Gregor et al. [[pdf]](http://arxiv.org/pdf/1502.04623)
- Rich feature hierarchies for accurate object detection and semantic segmentation (2014), R. Girshick et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Learning and transferring mid-Level image representations using convolutional neural networks (2014), M. Oquab et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification (2014), Y. Taigman et al. *(Facebook)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) :sparkles:
- Decaf: A deep convolutional activation feature for generic visual recognition (2013), J. Donahue et al. [[pdf]](http://arxiv.org/pdf/1310.1531) :sparkles:
- Learning hierarchical features for scene labeling (2013), C. Farabet et al. *(LeCun)* [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)
- Learning mid-level features for recognition (2010), Y. Boureau *(LeCun)* [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)

### Caption
- Show, attend and tell: Neural image caption generation with visual attention (2015), K. Xu et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1502.03044) :sparkles:
- Show and tell: A neural image caption generator (2015), O. Vinyals et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) :sparkles:
- Long-term recurrent convolutional networks for visual recognition and description (2015), J. Donahue et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Deep visual-semantic alignments for generating image descriptions (2015), A. Karpathy and L. Fei-Fei [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) :sparkles:

### Video / Human Activity
- Large-scale video classification with convolutional neural networks (2014), A. Karpathy et al. *(FeiFei)* [[pdf]](http://vision.stanford.edu/pdf/karpathy14.pdf) :sparkles:
- DeepPose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
- Two-stream convolutional networks for action recognition in videos (2014), K. Simonyan et al. [[pdf]](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
- A survey on human activity recognition using wearable sensors (2013), O. Lara and M. Labrador [[pdf]](http://romisatriawahono.net/lecture/rm/survey/computer%20vision/Lara%20-%20Human%20Activity%20Recognition%20-%202013.pdf)
- 3D convolutional neural networks for human action recognition (2013), S. Ji et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_JiXYY10.pdf)
- Action recognition with improved trajectories (2013), H. Wang and C. Schmid [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf)
- Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis (2011), Q. Le et al. [[pdf]](http://robotics.stanford.edu/~wzou/cvpr_LeZouYeungNg11.pdf)

### Word Embedding
- Glove: Global vectors for word representation (2014), J. Pennington et al. [[pdf]](http://anthology.aclweb.org/D/D14/D14-1162.pdf) :sparkles:
- Distributed representations of sentences and documents (2014), Q. Le and T. Mikolov [[pdf]](http://arxiv.org/pdf/1405.4053) *(Google)* :sparkles:
- Distributed representations of words and phrases and their compositionality (2013), T. Mikolov et al. *(Google)* [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) :sparkles:
- Efficient estimation of word representations in vector space (2013), T. Mikolov et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1301.3781) :sparkles:
- Word representations: a simple and general method for semi-supervised learning (2010), J. Turian *(Bengio)* [[pdf]](http://www.anthology.aclweb.org/P/P10/P10-1040.pdf)

### Machine Translation / QnA
- Towards ai-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](http://arxiv.org/pdf/1502.05698)
- Neural machine translation by jointly learning to align and translate (2014), D. Bahdanau et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1409.0473) :sparkles:
- Sequence to sequence learning with neural networks (2014), I. Sutskever et al. [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :sparkles:
- Learning phrase representations using RNN encoder-decoder for statistical machine translation (2014), K. Cho et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1406.1078)
- A convolutional neural network for modelling sentences (2014), N. Kalchbrenner et al. [[pdf]](http://arxiv.org/pdf/1404.2188v1)
- Convolutional neural networks for sentence classification (2014), Y. Kim [[pdf]](http://arxiv.org/pdf/1408.5882)
- The stanford coreNLP natural language processing toolkit (2014), C. Manning et al. [[pdf]](http://www.surdeanu.info/mihai/papers/acl2014-corenlp.pdf) :sparkles:
- Recursive deep models for semantic compositionality over a sentiment treebank (2013), R. Socher et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf) :sparkles:
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](http://arxiv.org/pdf/1103.0398) :sparkles:
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)

### Speech / Etc.
- Automatic speech recognition - A deep learning approach (Book, 2015), D. Yu and L. Deng *(Microsoft)* [[html]](http://www.springer.com/us/book/9781447157786)
- Speech recognition with deep recurrent neural networks (2013), A. Graves *(Hinton)* [[pdf]](http://arxiv.org/pdf/1303.5778.pdf)
- Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups (2012), G. Hinton et al. [[pdf]](http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf) :sparkles:
- Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition (2012) G. Dahl et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf) :sparkles:
- Acoustic modeling using deep belief networks (2012), A. Mohamed et al. *(Hinton)* [[pdf]](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)


### RL / Robotics
- Mastering the game of Go with deep neural networks and tree search (2016), D. Silver et al. *(DeepMind)* [[pdf]](Mastering the game of Go with deep neural networks and tree search) :sparkles:
- Human-level control through deep reinforcement learning (2015), V. Mnih et al. *(DeepMind)* [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf) :sparkles:
- Deep learning for detecting robotic grasps (2015), I. Lenz et al. [[pdf]](http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
- Playing atari with deep reinforcement learning (2013), V. Mnih et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1312.5602.pdf))


### Unsupervised
- Generative adversarial nets (2014), I. Goodfellow et al. *(Bengio)* [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- Auto-encoding variational Bayes (2013), D. Kingma and M. Welling [[pdf]](http://arxiv.org/pdf/1312.6114)
- Building high-level features using large scale unsupervised learning (2013), Q. Le et al. [[pdf]](http://arxiv.org/pdf/1112.6209) :sparkles:
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)

### Hardware / Software
- TensorFlow: Large-scale machine learning on heterogeneous distributed systems (2016), M. Abadi et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1603.04467)
- Theano: A Python framework for fast computation of mathematical expressions, R. Al-Rfou et al. *(Bengio)*
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](http://arxiv.org/pdf/1408.5093) :sparkles:


### Papers Worth Reading
*Newly released papers which do not meet the criteria but worth reading*
- Identity Mappings in Deep Residual Networks (2016), K. He et al. *(Microsoft)* [[pdf]](https://arxiv.org/pdf/1603.05027v2.pdf)
- Adversarially learned inference (2016), V. Dumoulin et al. [[web]](https://ishmaelbelghazi.github.io/ALI/)[[pdf]](https://arxiv.org/pdf/1606.00704v1)
- Understanding convolutional neural networks (2016), J. Koushik [[pdf]](https://arxiv.org/pdf/1605.09081v1)
- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size (2016), F. Iandola et al. [[pdf]](http://arxiv.org/pdf/1602.07360)
- Learning to compose neural networks for question answering (2016), J. Andreas et al. [[pdf]](http://arxiv.org/pdf/1601.01705)
- Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection (2016) *(Google)*, S. Levine et al. [[pdf]](http://arxiv.org/pdf/1603.02199v3)
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Eie: Efficient inference engine on compressed deep neural network (2016), S. Han et al. [[pdf]](http://arxiv.org/pdf/1602.01528)
- Adaptive Computation Time for Recurrent Neural Networks (2016), A. Graves [[pdf]](http://arxiv.org/pdf/1603.08983)
- Pixel recurrent neural networks (2016), A. van den Oord et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)
- LSTM: A search space odyssey (2015), K. Greff et al. [[pdf]](http://arxiv.org/pdf/1503.04069)
- Training very deep networks (2015), R. Srivastava et al. [[pdf]](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)

### Classic Papers
*Classic papers (1997~2009) which cause the advent of deep learning era*
- Learning deep architectures for AI (2009), Y. Bengio. [[pdf]](http://sanghv.com/download/soft/machine%20learning,%20artificial%20intelligence,%20mathematics%20ebooks/ML/learning%20deep%20architectures%20for%20AI%20(2009).pdf)
- Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations (2009), H. Lee et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf)
- Greedy layer-wise training of deep networks (2007), Y. Bengio et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- Reducing the dimensionality of data with neural networks, G. Hinton and R. Salakhutdinov. [[pdf]](http://homes.mpimf-heidelberg.mpg.de/~mhelmsta/pdf/2006%20Hinton%20Salakhudtkinov%20Science.pdf)
- A fast learning algorithm for deep belief nets (2006), G. Hinton et al. [[pdf]](http://nuyoo.utm.mx/~jjf/rna/A8%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)
- Gradient-based learning applied to document recognition (1998), Y. LeCun et al. [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) 
- Long short-term memory (1997), S. Hochreiter and J. Schmidhuber. [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735)

### Distinguished Researchers
*Distinguished deep learning researchers who have published +3 (:sparkles: +6) papers which are on the awesome list*
 (The papers in *Hardware / Software*, *Papers Worth Reading*, *Classic Papers* sections are excluded in counting.)

- [Jian Sun](https://scholar.google.ca/citations?user=ALVSZAYAAAAJ), *Microsoft Research* :sparkles:
- [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), *Google, University of Toronto* :sparkles:
- [Quoc Le](https://scholar.google.ca/citations?user=vfT6-XIAAAAJ), *Google* :sparkles:
- [Yann LeCun](https://scholar.google.ca/citations?user=WLN3QrAAAAAJ), *Facebook, New York University* :sparkles:
- [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), *University of Montreal* :sparkles:
- [Aaron Courville](https://scholar.google.ca/citations?user=km6CP8cAAAAJ), *University of Montreal*
- [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ), *Google DeepMind*
- [Andrej Karpathy](https://scholar.google.ca/citations?hl=en&user=l8WuQJgAAAAJ), *Stanford University*
- [Andrew Ng](https://scholar.google.ca/citations?user=JgDKULMAAAAJ), *Baidu*
- [Andrew Zisserman](https://scholar.google.ca/citations?user=UZ5wscMAAAAJ), *University of Oxford*
- [Christopher Manning](https://scholar.google.ca/citations?hl=en&user=1zmDOdwAAAAJ), *Stanford University*
- [David Silver](https://scholar.google.ca/citations?user=-8DNE4UAAAAJ), *Google DeepMind*
- [Dong Yu](https://scholar.google.ca/citations?hl=en&user=tMY31_gAAAAJ), *Microsoft Research*
- [Ross Girshick](https://scholar.google.ca/citations?user=W8VIEZgAAAAJ), *Facebook*
- [Kaiming He](https://scholar.google.ca/citations?user=DhtAFkwAAAAJ), *Microsoft Research* 
- [Karen Simonyan](https://scholar.google.ca/citations?user=L7lMQkQAAAAJ), *Google DeepMind*
- [Kyunghyun Cho](https://scholar.google.ca/citations?user=0RAmmIAAAAAJ), *New York University*
- [Honglak Lee](https://scholar.google.ca/citations?hl=en&user=fmSHtE8AAAAJ), *University of Michigan*
- [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ), *Google*
- [Ilya Sutskever](https://scholar.google.ca/citations?user=x04W_mMAAAAJ), *OpenAI* 
- [Jeff Dean](https://scholar.google.ca/citations?user=NMS69lQAAAAJ), *Google*,
- [Jeff Donahue](https://scholar.google.ca/citations?hl=en&user=UfbuDH8AAAAJ), *U.C. Berkeley*
- [Juergen Schmidhuber](https://scholar.google.ca/citations?user=gLnCTgIAAAAJ), *Swiss AI Lab IDSIA*
- [Li Fei-Fei](https://scholar.google.ca/citations?hl=en&user=rDfyQnIAAAAJ), *Stanford University*
- [Oriol Vinyals](https://scholar.google.ca/citations?user=NkzyCvUAAAAJ), *Google DeepMind*
- [Pascal Vincent](https://scholar.google.ca/citations?user=WBCKQMsAAAAJ), *University of Montreal*
- [Rob Fergus](https://scholar.google.ca/citations?user=GgQ9GEkAAAAJ), *Facebook, New York University*
- [Ruslan Salakhutdinov](https://scholar.google.ca/citations?user=ITZ1e7MAAAAJ), *CMU*
- [Tomas Mikolov](https://scholar.google.ca/citations?hl=en&user=oBu8kMMAAAAJ), *Facebook*
- [Trevor Darrell](https://scholar.google.ca/citations?user=bh-uRFMAAAAJ), *U.C. Berkeley* 

## Acknowledgement

Thank you for all your contributions. Please make sure to read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) before you make a pull request.

You can follow my [facebook page](https://www.facebook.com/terryum.io/) or [google plus](https://plus.google.com/+TerryTaeWoongUm/) to get useful information about machine learning and robotics. If you want to have a talk with me, please send me a message to my [facebook page](https://www.facebook.com/terryum.io/).

You can also check out my [blog](http://terryum.io/) where I share my thoughts on my research area (deep learning for human/robot motions). I got some thoughts while making this list and summerized them in a blog post, ["Some trends of recent deep learning researches"](http://terryum.io/ml_theory/2016/06/05/DeepLearningPapers/).

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Terry T. Um](https://www.facebook.com/terryum.io/) has waived all copyright and related or neighboring rights to this work.
