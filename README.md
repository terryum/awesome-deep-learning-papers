# Awesome - Most Cited Deep Learning Papers

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of the most cited deep learning papers (since 2012)

We believe that there exist *classic* deep learning papers which are worth reading regardless of their application domain. Rather than providing overwhelming amount of papers, We would like to provide a *curated list* of the awesome deep learning papers which are considered as *must-reads* in certain research domains.

## Background

Before this list, there exist other *awesome deep learning lists*, for example, [Deep Vision](https://github.com/kjw0612/awesome-deep-vision) and [Awesome Recurrent Neural Networks](https://github.com/kjw0612/awesome-rnn). Also, after this list comes out, another awesome list for deep learning beginners, called [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap), has been created and loved by many deep learning researchers.

Although the *Roadmap List* includes lots of important deep learning papers, it feels overwhelming for me to read them all. As I mentioned in the introduction, I believe that seminal works can give us lessons regardless of their application domain. Thus, I would like to introduce **top 100 deep learning papers** here as a good starting point of overviewing deep learning researches.

To get the news for newly released papers everyday, follow my [twitter](https://twitter.com/TerryUm_ML) or [facebook page](https://www.facebook.com/terryum.io/)! 

## Awesome list criteria

1. A list of **top 100 deep learning papers** published from 2012 to 2016 is suggested.
2. If a paper is added to the list, another paper (usually from *More Papers from 2016" section) should be removed to keep top 100 papers. (Thus, removing papers is also important contributions as well as adding papers)
3. Papers that are important, but failed to be included in the list, will be listed in *More than Top 100* section.
4. Please refer to *New Papers* and *Old Papers* sections for the papers published in recent 6 months or before 2012.

*(Citation criteria)*
- **< 6 months** : *New Papers* (by discussion)
- **2016** :  +60 citations or "More Papers from 2016"
- **2015** :  +200 citations
- **2014** :  +400 citations
- **2013** :  +600 citations
- **2012** :  +800 citations
- **~2012** : *Old Papers* (by discussion)

Please note that we prefer seminal deep learning papers that can be applied to various researches rather than application papers. For that reason, some papers that meet the criteria may not be accepted while others can be. It depends on the impact of the paper, applicability to other researches scarcity of the research domain, and so on.

**We need your contributions!**

If you have any suggestions (missing papers, new papers, key researchers or typos), please feel free to edit and pull a request.
(Please read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) for further instructions, though just letting me know the title of papers can also be a big contribution to us.)

(Update) You can download all top-100 papers with [this](https://github.com/terryum/awesome-deep-learning-papers/blob/master/fetch_papers.py) and collect all authors' names with [this](https://github.com/terryum/awesome-deep-learning-papers/blob/master/get_authors.py). Also, [bib file](https://github.com/terryum/awesome-deep-learning-papers/blob/master/top100papers.bib) for all top-100 papers are available. Thanks, doodhwala, [Sven](https://github.com/sunshinemyson) and [grepinsight](https://github.com/grepinsight)!

+ Can anyone contribute the code for obtaining the statistics of the authors of Top-100 papers?


## Contents

* [Understanding / Generalization / Transfer](#understanding--generalization--transfer)
* [Optimization / Training Techniques](#optimization--training-techniques)
* [Unsupervised / Generative Models](#unsupervised--generative-models)
* [Convolutional Network Models](#convolutional-neural-network-models)
* [Image Segmentation / Object Detection](#image-segmentation--object-detection)
* [Image / Video / Etc](#image--video--etc)
* [Natural Language Processing / RNNs](#natural-language-processing--rnns)
* [Speech / Other Domain](#speech--other-domain)
* [Reinforcement Learning / Robotics](#reinforcement-learning--robotics)
* [More Papers from 2016](#more-papers-from-2016)

*(More than Top 100)*

* [New Papers](#new-papers) : Less than 6 months
* [Old Papers](#old-papers) : Before 2012
* [HW / SW / Dataset](#hw--sw--dataset) : Technical reports
* [Book / Survey / Review](#book--survey--review)
* [Video Lectures / Tutorials / Blogs](#video-lectures--tutorials--blogs)
* [Appendix: More than Top 100](#appendix-more-than-top-100) : More papers not in the list

* * *

### Understanding / Generalization / Transfer
- **Distilling the knowledge in a neural network** (2015), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1503.02531)
- **Deep neural networks are easily fooled: High confidence predictions for unrecognizable images** (2015), A. Nguyen et al. [[pdf]](http://arxiv.org/pdf/1412.1897)
- **How transferable are features in deep neural networks?** (2014), J. Yosinski et al. [[pdf]](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- **CNN features off-the-Shelf: An astounding baseline for recognition** (2014), A. Razavian et al. [[pdf]](http://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)
- **Learning and transferring mid-Level image representations using convolutional neural networks** (2014), M. Oquab et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)
- **Visualizing and understanding convolutional networks** (2014), M. Zeiler and R. Fergus [[pdf]](http://arxiv.org/pdf/1311.2901)
- **Decaf: A deep convolutional activation feature for generic visual recognition** (2014), J. Donahue et al. [[pdf]](http://arxiv.org/pdf/1310.1531)

<!---[Key researchers]  [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Jason Yosinski](https://scholar.google.ca/citations?hl=en&user=gxL1qj8AAAAJ) -->

### Optimization / Training Techniques
- **Training very deep networks** (2015), R. Srivastava et al. [[pdf]](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)
- **Batch normalization: Accelerating deep network training by reducing internal covariate shift** (2015), S. Loffe and C. Szegedy [[pdf]](http://arxiv.org/pdf/1502.03167)
- **Delving deep into rectifiers: Surpassing human-level performance on imagenet classification** (2015), K. He et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
- **Dropout: A simple way to prevent neural networks from overfitting** (2014), N. Srivastava et al. [[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- **Adam: A method for stochastic optimization** (2014), D. Kingma and J. Ba [[pdf]](http://arxiv.org/pdf/1412.6980)
- **Improving neural networks by preventing co-adaptation of feature detectors** (2012), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1207.0580.pdf)
- **Random search for hyper-parameter optimization** (2012) J. Bergstra and Y. Bengio [[pdf]](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a)

<!---[Key researchers] [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Christian Szegedy](https://scholar.google.ca/citations?hl=en&user=3QeF7mAAAAAJ), [Sergey Ioffe](https://scholar.google.ca/citations?user=S5zOyIkAAAAJ), [Kaming He](https://scholar.google.ca/citations?hl=en&user=DhtAFkwAAAAJ), [Diederik P. Kingma](https://scholar.google.ca/citations?hl=en&user=yyIoQu4AAAAJ)-->

### Unsupervised / Generative Models
- **Pixel recurrent neural networks** (2016), A. Oord et al. [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)
- **Improved techniques for training GANs** (2016), T. Salimans et al. [[pdf]](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)
- **Unsupervised representation learning with deep convolutional generative adversarial networks** (2015), A. Radford et al. [[pdf]](https://arxiv.org/pdf/1511.06434v2)
- **DRAW: A recurrent neural network for image generation** (2015), K. Gregor et al. [[pdf]](http://arxiv.org/pdf/1502.04623)
- **Generative adversarial nets** (2014), I. Goodfellow et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- **Auto-encoding variational Bayes** (2013), D. Kingma and M. Welling [[pdf]](http://arxiv.org/pdf/1312.6114)
- **Building high-level features using large scale unsupervised learning** (2013), Q. Le et al. [[pdf]](http://arxiv.org/pdf/1112.6209)

<!---[Key researchers] [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ), [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ)-->
### Convolutional Neural Network Models
- **Rethinking the inception architecture for computer vision** (2016), C. Szegedy et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
- **Inception-v4, inception-resnet and the impact of residual connections on learning** (2016), C. Szegedy et al. [[pdf]](http://arxiv.org/pdf/1602.07261)
- **Identity Mappings in Deep Residual Networks** (2016), K. He et al. [[pdf]](https://arxiv.org/pdf/1603.05027v2.pdf)
- **Deep residual learning for image recognition** (2016), K. He et al. [[pdf]](http://arxiv.org/pdf/1512.03385)
- **Spatial transformer network** (2015), M. Jaderberg et al., [[pdf]](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)
- **Going deeper with convolutions** (2015), C. Szegedy et al.  [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- **Very deep convolutional networks for large-scale image recognition** (2014), K. Simonyan and A. Zisserman [[pdf]](http://arxiv.org/pdf/1409.1556)
- **Return of the devil in the details: delving deep into convolutional nets** (2014), K. Chatfield et al. [[pdf]](http://arxiv.org/pdf/1405.3531)
- **OverFeat: Integrated recognition, localization and detection using convolutional networks** (2013), P. Sermanet et al. [[pdf]](http://arxiv.org/pdf/1312.6229)
- **Maxout networks** (2013), I. Goodfellow et al. [[pdf]](http://arxiv.org/pdf/1302.4389v4)
- **Network in network** (2013), M. Lin et al. [[pdf]](http://arxiv.org/pdf/1312.4400)
- **ImageNet classification with deep convolutional neural networks** (2012), A. Krizhevsky et al. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

<!---[Key researchers]  [Christian Szegedy](https://scholar.google.ca/citations?hl=en&user=3QeF7mAAAAAJ), [Kaming He](https://scholar.google.ca/citations?hl=en&user=DhtAFkwAAAAJ), [Shaoqing Ren](https://scholar.google.ca/citations?hl=en&user=AUhj438AAAAJ), [Jian Sun](https://scholar.google.ca/citations?hl=en&user=ALVSZAYAAAAJ), [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), [Yann LeCun](https://scholar.google.ca/citations?hl=en&user=WLN3QrAAAAAJ)-->

### Image: Segmentation / Object Detection
- **You only look once: Unified, real-time object detection** (2016), J. Redmon et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
- **Fully convolutional networks for semantic segmentation** (2015), J. Long et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** (2015), S. Ren et al. [[pdf]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
- **Fast R-CNN** (2015), R. Girshick [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
- **Rich feature hierarchies for accurate object detection and semantic segmentation** (2014), R. Girshick et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
- **Spatial pyramid pooling in deep convolutional networks for visual recognition** (2014), K. He et al. [[pdf]](http://arxiv.org/pdf/1406.4729)
- **Semantic image segmentation with deep convolutional nets and fully connected CRFs**, L. Chen et al. [[pdf]](https://arxiv.org/pdf/1412.7062)
- **Learning hierarchical features for scene labeling** (2013), C. Farabet et al. [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)

<!---[Key researchers]  [Ross Girshick](https://scholar.google.ca/citations?hl=en&user=W8VIEZgAAAAJ), [Jeff Donahue](https://scholar.google.ca/citations?hl=en&user=UfbuDH8AAAAJ), [Trevor Darrell](https://scholar.google.ca/citations?hl=en&user=bh-uRFMAAAAJ)-->

### Image / Video / Etc
- **Image Super-Resolution Using Deep Convolutional Networks** (2016), C. Dong et al. [[pdf]](https://arxiv.org/pdf/1501.00092v3.pdf)
- **A neural algorithm of artistic style** (2015), L. Gatys et al. [[pdf]](https://arxiv.org/pdf/1508.06576)
- **Deep visual-semantic alignments for generating image descriptions** (2015), A. Karpathy and L. Fei-Fei [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)
- **Show, attend and tell: Neural image caption generation with visual attention** (2015), K. Xu et al. [[pdf]](http://arxiv.org/pdf/1502.03044)
- **Show and tell: A neural image caption generator** (2015), O. Vinyals et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
- **Long-term recurrent convolutional networks for visual recognition and description** (2015), J. Donahue et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)
- **VQA: Visual question answering** (2015), S. Antol et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)
- **DeepFace: Closing the gap to human-level performance in face verification** (2014), Y. Taigman et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf):
- **Large-scale video classification with convolutional neural networks** (2014), A. Karpathy et al. [[pdf]](http://vision.stanford.edu/pdf/karpathy14.pdf)
- **Two-stream convolutional networks for action recognition in videos** (2014), K. Simonyan et al. [[pdf]](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
- **3D convolutional neural networks for human action recognition** (2013), S. Ji et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_JiXYY10.pdf)

<!---[Key researchers]  [Oriol Vinyals](https://scholar.google.ca/citations?user=NkzyCvUAAAAJ), [Andrej Karpathy](https://scholar.google.ca/citations?user=l8WuQJgAAAAJ)-->

<!---[Key researchers]  [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ)-->

### Natural Language Processing / RNNs
- **Neural Architectures for Named Entity Recognition** (2016), G. Lample et al. [[pdf]](http://aclweb.org/anthology/N/N16/N16-1030.pdf)
- **Exploring the limits of language modeling** (2016), R. Jozefowicz et al. [[pdf]](http://arxiv.org/pdf/1602.02410)
- **Teaching machines to read and comprehend** (2015), K. Hermann et al. [[pdf]](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf)
- **Effective approaches to attention-based neural machine translation** (2015), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1508.04025)
- **Conditional random fields as recurrent neural networks** (2015), S. Zheng and S. Jayasumana. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf)
- **Memory networks** (2014), J. Weston et al. [[pdf]](https://arxiv.org/pdf/1410.3916)
- **Neural turing machines** (2014), A. Graves et al. [[pdf]](https://arxiv.org/pdf/1410.5401)
- **Neural machine translation by jointly learning to align and translate** (2014), D. Bahdanau et al. [[pdf]](http://arxiv.org/pdf/1409.0473)
- **Sequence to sequence learning with neural networks** (2014), I. Sutskever et al. [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- **Learning phrase representations using RNN encoder-decoder for statistical machine translation** (2014), K. Cho et al. [[pdf]](http://arxiv.org/pdf/1406.1078)
- **A convolutional neural network for modeling sentences** (2014), N. Kalchbrenner et al. [[pdf]](http://arxiv.org/pdf/1404.2188v1)
- **Convolutional neural networks for sentence classification** (2014), Y. Kim [[pdf]](http://arxiv.org/pdf/1408.5882)
- **Glove: Global vectors for word representation** (2014), J. Pennington et al. [[pdf]](http://anthology.aclweb.org/D/D14/D14-1162.pdf)
- **Distributed representations of sentences and documents** (2014), Q. Le and T. Mikolov [[pdf]](http://arxiv.org/pdf/1405.4053)
- **Distributed representations of words and phrases and their compositionality** (2013), T. Mikolov et al. [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- **Efficient estimation of word representations in vector space** (2013), T. Mikolov et al.  [[pdf]](http://arxiv.org/pdf/1301.3781)
- **Recursive deep models for semantic compositionality over a sentiment treebank** (2013), R. Socher et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)
- **Generating sequences with recurrent neural networks** (2013), A. Graves. [[pdf]](https://arxiv.org/pdf/1308.0850)

<!---[Key researchers]  [Kyunghyun Cho](https://scholar.google.ca/citations?user=0RAmmIAAAAAJ), [Oriol Vinyals](https://scholar.google.ca/citations?user=NkzyCvUAAAAJ), [Richard Socher](https://scholar.google.ca/citations?hl=en&user=FaOcyfMAAAAJ), [Tomas Mikolov](https://scholar.google.ca/citations?user=oBu8kMMAAAAJ), [Christopher D. Manning](https://scholar.google.ca/citations?user=1zmDOdwAAAAJ), [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ)-->

### Speech / Other Domain
- **End-to-end attention-based large vocabulary speech recognition** (2016), D. Bahdanau et al. [[pdf]](https://arxiv.org/pdf/1508.04395)
- **Deep speech 2: End-to-end speech recognition in English and Mandarin** (2015), D. Amodei et al. [[pdf]](https://arxiv.org/pdf/1512.02595)
- **Speech recognition with deep recurrent neural networks** (2013), A. Graves [[pdf]](http://arxiv.org/pdf/1303.5778.pdf)
- **Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups** (2012), G. Hinton et al. [[pdf]](http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf)
- **Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition** (2012) G. Dahl et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf)
- **Acoustic modeling using deep belief networks** (2012), A. Mohamed et al. [[pdf]](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)

<!---[Key researchers]  [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ), [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), [Dong Yu](https://scholar.google.ca/citations?hl=en&user=tMY31_gAAAAJ)-->

### Reinforcement Learning / Robotics
- **End-to-end training of deep visuomotor policies** (2016), S. Levine et al. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/source/15-522.pdf)
- **Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection** (2016), S. Levine et al. [[pdf]](https://arxiv.org/pdf/1603.02199)
- **Asynchronous methods for deep reinforcement learning** (2016), V. Mnih et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf)
- **Deep Reinforcement Learning with Double Q-Learning** (2016), H. Hasselt et al. [[pdf]](https://arxiv.org/pdf/1509.06461.pdf )
- **Mastering the game of Go with deep neural networks and tree search** (2016), D. Silver et al. [[pdf]](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)
- **Continuous control with deep reinforcement learning** (2015), T. Lillicrap et al. [[pdf]](https://arxiv.org/pdf/1509.02971)
- **Human-level control through deep reinforcement learning** (2015), V. Mnih et al. [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf)
- **Deep learning for detecting robotic grasps** (2015), I. Lenz et al. [[pdf]](http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
- **Playing atari with deep reinforcement learning** (2013), V. Mnih et al. [[pdf]](http://arxiv.org/pdf/1312.5602.pdf))

<!---[Key researchers]  [Sergey Levine](https://scholar.google.ca/citations?user=8R35rCwAAAAJ), [Volodymyr Mnih](https://scholar.google.ca/citations?hl=en&user=rLdfJ1gAAAAJ), [David Silver](https://scholar.google.ca/citations?user=-8DNE4UAAAAJ)-->

### More Papers from 2016
- **Layer Normalization** (2016), J. Ba et al. [[pdf]](https://arxiv.org/pdf/1607.06450v1.pdf)
- **Learning to learn by gradient descent by gradient descent** (2016), M. Andrychowicz et al. [[pdf]](http://arxiv.org/pdf/1606.04474v1)
- **Domain-adversarial training of neural networks** (2016), Y. Ganin et al. [[pdf]](http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf)
- **WaveNet: A Generative Model for Raw Audio** (2016), A. Oord et al. [[pdf]](https://arxiv.org/pdf/1609.03499v2) [[web]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- **Colorful image colorization** (2016), R. Zhang et al. [[pdf]](https://arxiv.org/pdf/1603.08511)
- **Generative visual manipulation on the natural image manifold** (2016), J. Zhu et al. [[pdf]](https://arxiv.org/pdf/1609.03552)
- **Texture networks: Feed-forward synthesis of textures and stylized images** (2016), D Ulyanov et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/ulyanov16.pdf)
- **SSD: Single shot multibox detector** (2016), W. Liu et al. [[pdf]](https://arxiv.org/pdf/1512.02325)
- **SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size** (2016), F. Iandola et al. [[pdf]](http://arxiv.org/pdf/1602.07360)
- **Eie: Efficient inference engine on compressed deep neural network** (2016), S. Han et al. [[pdf]](http://arxiv.org/pdf/1602.01528)
- **Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1** (2016), M. Courbariaux et al. [[pdf]](https://arxiv.org/pdf/1602.02830)
- **Dynamic memory networks for visual and textual question answering** (2016), C. Xiong et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/xiong16.pdf)
- **Stacked attention networks for image question answering** (2016), Z. Yang et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf)
- **Hybrid computing using a neural network with dynamic external memory** (2016), A. Graves et al. [[pdf]](https://www.gwern.net/docs/2016-graves.pdf)
- **Google's neural machine translation system: Bridging the gap between human and machine translation** (2016), Y. Wu et al. [[pdf]](https://arxiv.org/pdf/1609.08144)

* * *


### New papers
*Newly published papers (< 6 months) which are worth reading*
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017), Andrew G. Howard et al. [[pdf]](https://arxiv.org/pdf/1704.04861.pdf)
- Convolutional Sequence to Sequence Learning (2017), Jonas Gehring et al. [[pdf]](https://arxiv.org/pdf/1705.03122)
- A Knowledge-Grounded Neural Conversation Model (2017), Marjan Ghazvininejad et al. [[pdf]](https://arxiv.org/pdf/1702.01932)
- Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour (2017), Priya Goyal et al. [[pdf]](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h3.pdf)
- TACOTRON: Towards end-to-end speech synthesis (2017), Y. Wang et al. [[pdf]](https://arxiv.org/pdf/1703.10135.pdf)
- Deep Photo Style Transfer (2017), F. Luan et al. [[pdf]](http://arxiv.org/pdf/1703.07511v1.pdf)
- Evolution Strategies as a Scalable Alternative to Reinforcement Learning (2017), T. Salimans et al. [[pdf]](http://arxiv.org/pdf/1703.03864v1.pdf)
- Deformable Convolutional Networks (2017), J. Dai et al. [[pdf]](http://arxiv.org/pdf/1703.06211v2.pdf)
- Mask R-CNN (2017), K. He et al. [[pdf]](https://128.84.21.199/pdf/1703.06870)
- Learning to discover cross-domain relations with generative adversarial networks (2017), T. Kim et al. [[pdf]](http://arxiv.org/pdf/1703.05192v1.pdf) 
- Deep voice: Real-time neural text-to-speech (2017), S. Arik et al., [[pdf]](http://arxiv.org/pdf/1702.07825v2.pdf)
- PixelNet: Representation of the pixels, by the pixels, and for the pixels (2017), A. Bansal et al. [[pdf]](http://arxiv.org/pdf/1702.06506v1.pdf)
- Batch renormalization: Towards reducing minibatch dependence in batch-normalized models (2017), S. Ioffe. [[pdf]](https://arxiv.org/abs/1702.03275)
- Wasserstein GAN (2017), M. Arjovsky et al. [[pdf]](https://arxiv.org/pdf/1701.07875v1)
- Understanding deep learning requires rethinking generalization (2017), C. Zhang et al. [[pdf]](https://arxiv.org/pdf/1611.03530)
- Least squares generative adversarial networks (2016), X. Mao et al. [[pdf]](https://arxiv.org/abs/1611.04076v2)


### Old Papers
*Classic papers published before 2012*
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- Deep sparse rectifier neural networks (2011), X. Glorot et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](http://arxiv.org/pdf/1103.0398)
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- Learning mid-level features for recognition (2010), Y. Boureau [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)
- Why does unsupervised pre-training help deep learning (2010), D. Erhan et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- Learning deep architectures for AI (2009), Y. Bengio. [[pdf]](http://sanghv.com/download/soft/machine%20learning,%20artificial%20intelligence,%20mathematics%20ebooks/ML/learning%20deep%20architectures%20for%20AI%20(2009).pdf)
- Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations (2009), H. Lee et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf)
- Greedy layer-wise training of deep networks (2007), Y. Bengio et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- Reducing the dimensionality of data with neural networks, G. Hinton and R. Salakhutdinov. [[pdf]](http://homes.mpimf-heidelberg.mpg.de/~mhelmsta/pdf/2006%20Hinton%20Salakhudtkinov%20Science.pdf)
- A fast learning algorithm for deep belief nets (2006), G. Hinton et al. [[pdf]](http://nuyoo.utm.mx/~jjf/rna/A8%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)
- Gradient-based learning applied to document recognition (1998), Y. LeCun et al. [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- Long short-term memory (1997), S. Hochreiter and J. Schmidhuber. [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735)


### HW / SW / Dataset
-  SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016), Rajpurkar et al. [[pdf]](https://arxiv.org/pdf/1606.05250.pdf)
- OpenAI gym (2016), G. Brockman et al. [[pdf]](https://arxiv.org/pdf/1606.01540)
- TensorFlow: Large-scale machine learning on heterogeneous distributed systems (2016), M. Abadi et al. [[pdf]](http://arxiv.org/pdf/1603.04467)
- Theano: A Python framework for fast computation of mathematical expressions, R. Al-Rfou et al.
- Torch7: A matlab-like environment for machine learning, R. Collobert et al. [[pdf]](https://ronan.collobert.com/pub/matos/2011_torch7_nipsw.pdf)
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](http://arxiv.org/pdf/1409.0575)
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](http://arxiv.org/pdf/1408.5093)


### Book / Survey / Review
- On the Origin of Deep Learning (2017), H. Wang and Bhiksha Raj. [[pdf]](https://arxiv.org/pdf/1702.07800)
- Deep Reinforcement Learning: An Overview (2017), Y. Li, [[pdf]](http://arxiv.org/pdf/1701.07274v2.pdf)
- Neural Machine Translation and Sequence-to-sequence Models(2017): A Tutorial, G. Neubig. [[pdf]](http://arxiv.org/pdf/1703.01619v1.pdf)
- Neural Network and Deep Learning (Book, Jan 2017), Michael Nielsen. [[html]](http://neuralnetworksanddeeplearning.com/index.html)
- Deep learning (Book, 2016), Goodfellow et al. [[html]](http://www.deeplearningbook.org/)
- LSTM: A search space odyssey (2016), K. Greff et al. [[pdf]](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)
- Tutorial on Variational Autoencoders (2016), C. Doersch. [[pdf]](https://arxiv.org/pdf/1606.05908)
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[pdf]](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](http://arxiv.org/pdf/1404.7828)
- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](http://arxiv.org/pdf/1206.5538)

### Video Lectures / Tutorials / Blogs

*(Lectures)*
- CS231n, Convolutional Neural Networks for Visual Recognition, Stanford University [[web]](http://cs231n.stanford.edu/)
- CS224d, Deep Learning for Natural Language Processing, Stanford University [[web]](http://cs224d.stanford.edu/)
- Oxford Deep NLP 2017, Deep Learning for Natural Language Processing, University of Oxford [[web]](https://github.com/oxford-cs-deepnlp-2017/lectures)

*(Tutorials)*
- NIPS 2016 Tutorials, Long Beach [[web]](https://nips.cc/Conferences/2016/Schedule?type=Tutorial)
- ICML 2016 Tutorials, New York City [[web]](http://techtalks.tv/icml/2016/tutorials/)
- ICLR 2016 Videos, San Juan [[web]](http://videolectures.net/iclr2016_san_juan/)
- Deep Learning Summer School 2016, Montreal [[web]](http://videolectures.net/deeplearning2016_montreal/)
- Bay Area Deep Learning School 2016, Stanford [[web]](https://www.bayareadlschool.org/)

*(Blogs)*
- OpenAI [[web]](https://www.openai.com/)
- Distill [[web]](http://distill.pub/)
- Andrej Karpathy Blog [[web]](http://karpathy.github.io/)
- Colah's Blog [[Web]](http://colah.github.io/)
- WildML [[Web]](http://www.wildml.com/)
- FastML [[web]](http://www.fastml.com/)
- TheMorningPaper [[web]](https://blog.acolyer.org)

### Appendix: More than Top 100
*(2016)*
- A character-level decoder without explicit segmentation for neural machine translation (2016), J. Chung et al. [[pdf]](https://arxiv.org/pdf/1603.06147)
- Dermatologist-level classification of skin cancer with deep neural networks (2017), A. Esteva et al. [[html]](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html)
- Weakly supervised object localization with multi-fold multiple instance learning (2017), R. Gokberk et al. [[pdf]](https://arxiv.org/pdf/1503.00949)
- Brain tumor segmentation with deep neural networks (2017), M. Havaei et al. [[pdf]](https://arxiv.org/pdf/1505.03540)
- Professor Forcing: A New Algorithm for Training Recurrent Networks (2016), A. Lamb et al. [[pdf]](https://arxiv.org/pdf/1610.09038)
- Adversarially learned inference (2016), V. Dumoulin et al. [[web]](https://ishmaelbelghazi.github.io/ALI/)[[pdf]](https://arxiv.org/pdf/1606.00704v1)
- Understanding convolutional neural networks (2016), J. Koushik [[pdf]](https://arxiv.org/pdf/1605.09081v1)
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Adaptive computation time for recurrent neural networks (2016), A. Graves [[pdf]](http://arxiv.org/pdf/1603.08983)
- Densely connected convolutional networks (2016), G. Huang et al. [[pdf]](https://arxiv.org/pdf/1608.06993v1)
- Region-based convolutional networks for accurate object detection and segmentation (2016), R. Girshick et al. 
- Continuous deep q-learning with model-based acceleration (2016), S. Gu et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/gu16.pdf)
- A thorough examination of the cnn/daily mail reading comprehension task (2016), D. Chen et al. [[pdf]](https://arxiv.org/pdf/1606.02858)
- Achieving open vocabulary neural machine translation with hybrid word-character models, M. Luong and C. Manning. [[pdf]](https://arxiv.org/pdf/1604.00788)
- Very Deep Convolutional Networks for Natural Language Processing (2016), A. Conneau et al. [[pdf]](https://arxiv.org/pdf/1606.01781)
- Bag of tricks for efficient text classification (2016), A. Joulin et al. [[pdf]](https://arxiv.org/pdf/1607.01759)
- Efficient piecewise training of deep structured models for semantic segmentation (2016), G. Lin et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lin_Efficient_Piecewise_Training_CVPR_2016_paper.pdf)
- Learning to compose neural networks for question answering (2016), J. Andreas et al. [[pdf]](https://arxiv.org/pdf/1601.01705)
- Perceptual losses for real-time style transfer and super-resolution (2016), J. Johnson et al. [[pdf]](https://arxiv.org/pdf/1603.08155)
- Reading text in the wild with convolutional neural networks (2016), M. Jaderberg et al. [[pdf]](http://arxiv.org/pdf/1412.1842)
- What makes for effective detection proposals? (2016), J. Hosang et al. [[pdf]](https://arxiv.org/pdf/1502.05082)
- Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks (2016), S. Bell et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf).
- Instance-aware semantic segmentation via multi-task network cascades (2016), J. Dai et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)
- Conditional image generation with pixelcnn decoders (2016), A. van den Oord et al. [[pdf]](http://papers.nips.cc/paper/6527-tree-structured-reinforcement-learning-for-sequential-object-localization.pdf)
- Deep networks with stochastic depth (2016), G. Huang et al., [[pdf]](https://arxiv.org/pdf/1603.09382)
- Consistency and Fluctuations For Stochastic Gradient Langevin Dynamics (2016), Yee Whye Teh et al. [[pdf]](http://www.jmlr.org/papers/volume17/teh16a/teh16a.pdf)

*(2015)*
- Ask your neurons: A neural-based approach to answering questions about images (2015), M. Malinowski et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.pdf)
- Exploring models and data for image question answering (2015), M. Ren et al. [[pdf]](http://papers.nips.cc/paper/5640-stochastic-variational-inference-for-hidden-markov-models.pdf)
- Are you talking to a machine? dataset and methods for multilingual image question (2015), H. Gao et al. [[pdf]](http://papers.nips.cc/paper/5641-are-you-talking-to-a-machine-dataset-and-methods-for-multilingual-image-question.pdf)
- Mind's eye: A recurrent visual representation for image caption generation (2015), X. Chen and C. Zitnick. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Chen_Minds_Eye_A_2015_CVPR_paper.pdf)
- From captions to visual concepts and back (2015), H. Fang et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fang_From_Captions_to_2015_CVPR_paper.pdf).
- Towards AI-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](http://arxiv.org/pdf/1502.05698)
- Ask me anything: Dynamic memory networks for natural language processing (2015), A. Kumar et al. [[pdf]](http://arxiv.org/pdf/1506.07285)
- Unsupervised learning of video representations using LSTMs (2015), N. Srivastava et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/srivastava15.pdf)
- Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding (2015), S. Han et al. [[pdf]](https://arxiv.org/pdf/1510.00149)
- Improved semantic representations from tree-structured long short-term memory networks (2015), K. Tai et al. [[pdf]](https://arxiv.org/pdf/1503.00075)
- Character-aware neural language models (2015), Y. Kim et al. [[pdf]](https://arxiv.org/pdf/1508.06615)
- Grammar as a foreign language (2015), O. Vinyals et al. [[pdf]](http://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf)
- Trust Region Policy Optimization (2015), J. Schulman et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf)
- Beyond short snippents: Deep networks for video classification (2015) [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf)
- Learning Deconvolution Network for Semantic Segmentation (2015), H. Noh et al. [[pdf]](https://arxiv.org/pdf/1505.04366v1)
- Learning spatiotemporal features with 3d convolutional networks (2015), D. Tran et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)
- Understanding neural networks through deep visualization (2015), J. Yosinski et al. [[pdf]](https://arxiv.org/pdf/1506.06579)
- An Empirical Exploration of Recurrent Network Architectures (2015), R. Jozefowicz et al.  [[pdf]](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- Deep generative image models using a￼ laplacian pyramid of adversarial networks (2015), E.Denton et al. [[pdf]](http://papers.nips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf)
- Gated Feedback Recurrent Neural Networks (2015), J. Chung et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/chung15.pdf)
- Fast and accurate deep network learning by exponential linear units (ELUS) (2015), D. Clevert et al. [[pdf]](https://arxiv.org/pdf/1511.07289.pdf%5Cnhttp://arxiv.org/abs/1511.07289%5Cnhttp://arxiv.org/abs/1511.07289)
- Pointer networks (2015), O. Vinyals et al. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf)
- Visualizing and Understanding Recurrent Networks (2015), A. Karpathy et al. [[pdf]](https://arxiv.org/pdf/1506.02078)
- Attention-based models for speech recognition (2015), J. Chorowski et al. [[pdf]](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)
- End-to-end memory networks (2015), S. Sukbaatar et al. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- Describing videos by exploiting temporal structure (2015), L. Yao et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf)
- A neural conversational model (2015), O. Vinyals and Q. Le. [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)
- Improving distributional similarity with lessons learned from word embeddings, O. Levy et al. [[pdf]] (https://www.transacl.org/ojs/index.php/tacl/article/download/570/124)
- Transition-Based Dependency Parsing with Stack Long Short-Term Memory (2015), C. Dyer et al. [[pdf]](http://aclweb.org/anthology/P/P15/P15-1033.pdf)
- Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs (2015), M. Ballesteros et al. [[pdf]](http://aclweb.org/anthology/D/D15/D15-1041.pdf)
- Finding function in form: Compositional character models for open vocabulary word representation (2015), W. Ling et al. [[pdf]](http://aclweb.org/anthology/D/D15/D15-1176.pdf)


*(~2014)*
- DeepPose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
- Learning a Deep Convolutional Network for Image Super-Resolution (2014, C. Dong et al. [[pdf]](https://www.researchgate.net/profile/Chen_Change_Loy/publication/264552416_Lecture_Notes_in_Computer_Science/links/53e583e50cf25d674e9c280e.pdf)
- Recurrent models of visual attention (2014), V. Mnih et al. [[pdf]](http://arxiv.org/pdf/1406.6247.pdf)
- Empirical evaluation of gated recurrent neural networks on sequence modeling (2014), J. Chung et al. [[pdf]](https://arxiv.org/pdf/1412.3555)
- Addressing the rare word problem in neural machine translation (2014), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1410.8206)
- On the properties of neural machine translation: Encoder-decoder approaches (2014), K. Cho et. al.
- Recurrent neural network regularization (2014), W. Zaremba et al. [[pdf]](http://arxiv.org/pdf/1409.2329)
- Intriguing properties of neural networks (2014), C. Szegedy et al. [[pdf]](https://arxiv.org/pdf/1312.6199.pdf)
- Towards end-to-end speech recognition with recurrent neural networks (2014), A. Graves and N. Jaitly. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
- Scalable object detection using deep neural networks (2014), D. Erhan et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)
- On the importance of initialization and momentum in deep learning (2013), I. Sutskever et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_sutskever13.pdf)
- Regularization of neural networks using dropconnect (2013), L. Wan et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf)
- Learning Hierarchical Features for Scene Labeling (2013), C. Farabet et al. [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)
- Linguistic Regularities in Continuous Space Word Representations (2013), T. Mikolov et al. [[pdf]](http://www.aclweb.org/anthology/N13-1#page=784)
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)
- A Fast and Accurate Dependency Parser using Neural Networks. Chen and Manning. [[pdf]](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)



## Acknowledgement

Thank you for all your contributions. Please make sure to read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) before you make a pull request.

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Terry T. Um](https://www.facebook.com/terryum.io/) has waived all copyright and related or neighboring rights to this work.
