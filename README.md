# Awesome - Most Cited Deep Learning Papers

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of the most cited deep learning papers (since 2012)

We believe that there exist *classic* deep learning papers which are worth reading regardless of their application domain. Rather than providing overwhelming amount of papers, We would like to provide a *curated list* of the awewome deep learning papers (less than 150 papers) which are considered as *must-reads* in a certain researh domain.

## Backgroud

Before this list, there exist other *awesome deep learning lists*, for example, [Deep Vision](https://github.com/kjw0612/awesome-deep-vision) and [Awesome Recurrent Neural Networks](https://github.com/kjw0612/awesome-rnn). Also, after this list comes out, another awesome list for deep learning beginners. called [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap), has been created and loved by many deep learning researchers.

Although the *Roadmap List* includes lots of important deep learning papers, it feels overwhelming for me to read them all. As I mentioned in the introduction, I believe that seminal works can give us lessons regardless of their application domain. Thus, I would like to introduce **top 100 deep learning papers** here as a good starting point of overviewing deep learning researches.

## Awesome list criteria

1. A list of **top 100 deep learning papers** published from 2012 to 2016 is suggested.
2. If a paper is added to the list, another paper should be removed from the list to keep top 100 papers. (Thus, removing papers is also important contributions as well as adding papers)
3. Papers that are important, but failed to be included in the list, will be listed in *More than Top 100* section. 
4. Please refer to *New Papers* and *Old Papers* sections for the papers published in recent 6 months or before 2012.

- **< 6 months** : can be added to *New Papers* section by discussion
- **2016** :  +30 citations (Bold +60)
- **2015** :  +100 citations (Bold +200)
- **2014** :  +200 citations (Bold +400)
- **2013** :  +300 citations (Bold +600)
- **2012** :  +400 citations (Bold +800)
- **~2011** : can be added to *Old Papers* section by discussion

Please note that we prefer seminal deep learning papers that can be applied to various researches rather than application papers. For that reason, some papers that meet the critera may not be accepted while others can be. It depends on the impact of the paper, applicability to other researches scarcity of the research domain, and so on.

**We need your contributions!** 

If you have any suggestions (missing papers, new papers, key researchers or typos), please feel free to edit and pull a request.
(Please read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) for futher instructions, though just letting me know the title of papers can also be a big contribution to us.)

## Table of Contents

* [Understanding / Generalization / Transfer](#understanding--generalization--transfer)
* [Optimization / Training Techniques](#optimization--training-techniques)
* [Unsupervised / Generative Models](#unsupervised--generative-models)
* [Convolutional Network Models](#convolutional-neural-network-models)
* [Image Segmentation / Object Detection](#image-segmentation--object-detection)
* [Visual QnA / Captioning / Video / Etc](#visual-qna--captioning--video--etc)
* [Recurrent Neural Network Models](#recurrent-neural-network-models)
* [Natural Language Process](#natural-language-process)
* [Reinforcement Learning](#reinforcement-learning)
* [Speech / Other Domain](#speech--other-domain)

*(More than Top 100)*

* [New Papers](#new-papers) : < 6 months
* [Old Papers](#old-papers) : < 2012
* [More than Top 100](#more-than-top-100) : Important papers not included in the top 100 list
* [HW / SW / Dataset](#hw--sw--dataset) : Technical reports
* [Book / Survey / Review](#book--survey--review)
* [Video Lectures / Tutorials / Blogs](#video-lectures--tutorials--blogs) 

* * *

### Understanding / Generalization / Transfer
- Distilling the knowledge in a neural network (2015), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1503.02531) :sparkles:
- Deep neural networks are easily fooled: High confidence predictions for unrecognizable images (2015), A. Nguyen et al. [[pdf]](http://arxiv.org/pdf/1412.1897)
- How transferable are features in deep neural networks? (2014), J. Yosinski et al. [[pdf]](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- Return of the devil in the details: delving deep into convolutional nets (2014), K. Chatfield et al. [[pdf]](http://arxiv.org/pdf/1405.3531) :sparkles:
- Why does unsupervised pre-training help deep learning (2010), D. Erhan et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)

### Optimization / Training Techniques
- Deep networks with stochastic depth (2016), G. Huang et al., [[pdf]](https://arxiv.org/pdf/1603.09382)
- Batch normalization: Accelerating deep network training by reducing internal covariate shift (2015), S. Loffe and C. Szegedy *(Google)* [[pdf]](http://arxiv.org/pdf/1502.03167) :sparkles:
- Fast and accurate deep network learning by exponential linear units (ELUS) (2015), D. Clevert et al. [[pdf]]()
- Delving deep into rectifiers: Surpassing human-level performance on imagenet classification (2015), K. He et al. *(He)* [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) :sparkles:
- Training very deep networks (2015), R. Srivastava et al. [[pdf]](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)
- Recurrent neural network regularization (2014), W. Zaremba et al. [[pdf]](http://arxiv.org/pdf/1409.2329)
- Dropout: A simple way to prevent neural networks from overfitting (2014), N. Srivastava et al. *(Hinton)* [[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) :sparkles:
- Adam: A method for stochastic optimization (2014), D. Kingma and J. Ba [[pdf]](http://arxiv.org/pdf/1412.6980)
- Spatial pyramid pooling in deep convolutional networks for visual recognition (2014), K. He et al. [[pdf]](http://arxiv.org/pdf/1406.4729)  :sparkles:
- On the importance of initialization and momentum in deep learning (2013), I. Sutskever et al. *(Hinton)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_sutskever13.pdf)
- Regularization of neural networks using dropconnect (2013), L. Wan et al. *(LeCun)* [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf)
- Improving neural networks by preventing co-adaptation of feature detectors (2012), G. Hinton et al. [[pdf]](http://arxiv.org/pdf/1207.0580.pdf) :sparkles:
- Random search for hyper-parameter optimization (2012) J. Bergstra and Y. Bengio [[pdf]](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a)

### Unsupervised / Generative Models
- Generative visual manipulation on the natural image manifold (2016), J. Zhu et al. [[pdf]](https://arxiv.org/pdf/1609.03552)
- Conditional image generation with pixelcnn decoders (2016), A. van den Oord et al. [[pdf]](http://papers.nips.cc/paper/6527-tree-structured-reinforcement-learning-for-sequential-object-localization.pdf)
- Pixel recurrent neural networks (2016), A. van den Oord et al. [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)
- Unsupervised representation learning with deep convolutional generative adversarial networks (2015), A. Radford et al. [[pdf]](https://arxiv.org/pdf/1511.06434v2)
- DRAW: A recurrent neural network for image generation (2015), K. Gregor et al. [[pdf]](http://arxiv.org/pdf/1502.04623)
- CNN features off-the-Shelf: An astounding baseline for recognition (2014), A. Razavian et al. [[pdf]](http://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf) :sparkles:
- Generative adversarial nets (2014), I. Goodfellow et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- Intriguing properties of neural networks (2014), C. Szegedy et al. *(Sutskever, Goodfellow: Google)* [[pdf]](https://arxiv.org/pdf/1312.6199.pdf)
- Auto-encoding variational Bayes (2013), D. Kingma and M. Welling [[pdf]](http://arxiv.org/pdf/1312.6114)
- Building high-level features using large scale unsupervised learning (2013), Q. Le et al. [[pdf]](http://arxiv.org/pdf/1112.6209) :sparkles:
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)


### Convolutional Neural Network Models
- Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1 (2016), M. Courbariaux et al. [[pdf]](https://arxiv.org/pdf/1602.02830)
- Inception-v4, inception-resnet and the impact of residual connections on learning (2016), C. Szegedy et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1602.07261)
- Identity Mappings in Deep Residual Networks (2016), K. He et al. *(He)* [[pdf]](https://arxiv.org/pdf/1603.05027v2.pdf)
- Deep residual learning for image recognition (2016), K. He et al. *(He)* [[pdf]](http://arxiv.org/pdf/1512.03385) :sparkles:
- Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding (2015), S. Han et al. [[pdf]](https://arxiv.org/pdf/1510.00149)
- Going deeper with convolutions (2015), C. Szegedy et al. *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) :sparkles:
- An Empirical Exploration of Recurrent Network Architectures (2015), R. Jozefowicz et al. *Sutskever: Google* [[pdf]](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- Fully convolutional networks for semantic segmentation (2015), J. Long et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) :sparkles:
- Very deep convolutional networks for large-scale image recognition (2014), K. Simonyan and A. Zisserman [[pdf]](http://arxiv.org/pdf/1409.1556) :sparkles:
- OverFeat: Integrated recognition, localization and detection using convolutional networks (2014), P. Sermanet et al. *(LeCun)* [[pdf]](http://arxiv.org/pdf/1312.6229)
- Visualizing and understanding convolutional networks (2014), M. Zeiler and R. Fergus [[pdf]](http://arxiv.org/pdf/1311.2901) :sparkles:
- Maxout networks (2013), I. Goodfellow et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1302.4389v4)
- Network in network (2013), M. Lin et al. [[pdf]](http://arxiv.org/pdf/1312.4400)
- ImageNet classification with deep convolutional neural networks (2012), A. Krizhevsky et al. *(Hinton)* [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) :sparkles:
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf) :sparkles:
- Deep sparse rectifier neural networks (2011), X. Glorot et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)


### Image: Segmentation / Object Detection
- Instance-aware semantic segmentation via multi-task network cascades (2016), J. Dai et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)
- Efficient piecewise training of deep structured models for semantic segmentation (2016), G. Lin et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lin_Efficient_Piecewise_Training_CVPR_2016_paper.pdf)
- SSD: Single shot multibox detector (2016), W. Liu et al. [[pdf]](https://arxiv.org/pdf/1512.02325) 
- You only look once: Unified, real-time object detection (2016), J. Redmon et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
- Region-based convolutional networks for accurate object detection and segmentation (2016), R. Girshick et al. *(He)* [[pdf]](https://www.cs.berkeley.edu/~rbg/papers/pami/rcnn_pami.pdf)
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015), S. Ren et al. [[pdf]] (http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) :sparkles:
- Fast R-CNN (2015), R. Girshick *(He)* [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) :sparkles:
- Scalable object detection using deep neural networks (2014), D. Erhan et al. *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)
- Rich feature hierarchies for accurate object detection and semantic segmentation (2014), R. Girshick et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Semantic image segmentation with deep convolutional nets and fully connected CRFs, L. Chen et al. [[pdf]](https://arxiv.org/pdf/1412.7062)
- Deep neural networks for object detection (2013), C. Szegedy et al. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

### Visual QnA / Captioning / Video / Etc
- Brain tumor segmentation with deep neural networks (2017), M. Havaei et al. *(Bengio)* [[pdf]](https://arxiv.org/pdf/1505.03540)
- Weakly supervised object localization with multi-fold multiple instance learning (2017), R. Gokberk et al. [[pdf]](https://arxiv.org/pdf/1503.00949)
- Dynamic memory networks for visual and textual question answering (2016), C. Xiong et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/xiong16.pdf)
- Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks (2016), S. Bell et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf).
- Perceptual losses for real-time style transfer and super-resolution (2016), J. Johnson et al. [[pdf]](https://arxiv.org/pdf/1603.08155)
- Colorful image colorization (2016), R. Zhang et al. [[pdf]](https://arxiv.org/pdf/1603.08511) 
- What makes for effective detection proposals? (2016), J. Hosan et al. *(Facebook)* [[pdf]](https://arxiv.org/pdf/1502.05082)
- Image Super-Resolution Using Deep Convolutional Networks (2016), C. Dong et al. *(He)* [[pdf]](https://arxiv.org/pdf/1501.00092v3.pdf)  :sparkles:
- Reading text in the wild with convolutional neural networks (2016), M. Jaderberg et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1412.1842)
- A neural algorithm of artistic style (2015), L. Gatys et al. [[pdf]](https://arxiv.org/pdf/1508.06576)
- Learning Deconvolution Network for Semantic Segmentation (2015), H. Noh et al. [[pdf]](https://arxiv.org/pdf/1505.04366v1)
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](http://arxiv.org/pdf/1409.0575) :sparkles:
- Learning a Deep Convolutional Network for Image Super-Resolution (2014, C. Dong et al. [[pdf]](https://www.researchgate.net/profile/Chen_Change_Loy/publication/264552416_Lecture_Notes_in_Computer_Science/links/53e583e50cf25d674e9c280e.pdf)
- Learning and transferring mid-Level image representations using convolutional neural networks (2014), M. Oquab et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification (2014), Y. Taigman et al. *(Facebook)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) :sparkles:
- Decaf: A deep convolutional activation feature for generic visual recognition (2013), J. Donahue et al. [[pdf]](http://arxiv.org/pdf/1310.1531) :sparkles:
- Learning hierarchical features for scene labeling (2013), C. Farabet et al. *(LeCun)* [[pdf]](https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf)
- Learning mid-level features for recognition (2010), Y. Boureau *(LeCun)* [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)

- Mind's eye: A recurrent visual representation for image caption generation (2015), X. Chen and C. Zitnick. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Chen_Minds_Eye_A_2015_CVPR_paper.pdf)
- From captions to visual concepts and back (2015), H. Fang et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fang_From_Captions_to_2015_CVPR_paper.pdf).
- VQA: Visual question answering (2015), S. Antol et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)
- Towards ai-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. *(Mikolov: Facebook)* [[pdf]](http://arxiv.org/pdf/1502.05698)
- Ask me anything: Dynamic memory networks for natural language processing (2015), A. Kumar et al. [[pdf]](http://arxiv.org/pdf/1506.07285)
- A large annotated corpus for learning natural language inference (2015), S. Bowman et al. [[pdf]](http://arxiv.org/pdf/1508.05326)
- Show, attend and tell: Neural image caption generation with visual attention (2015), K. Xu et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1502.03044) :sparkles:
- Show and tell: A neural image caption generator (2015), O. Vinyals et al. *(Vinyals: Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) :sparkles:
- Long-term recurrent convolutional networks for visual recognition and description (2015), J. Donahue et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :sparkles:
- Deep visual-semantic alignments for generating image descriptions (2015), A. Karpathy and L. Fei-Fei [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) :sparkles:
- Deep captioning with multimodal recurrent neural networks (m-rnn) (2014), J. Mao et al. [[pdf]](https://arxiv.org/pdf/1412.6632)

- Beyond short snippents: Deep networks for video classification (2015) *(Vinyals: Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf)  :sparkles:
- Large-scale video classification with convolutional neural networks (2014), A. Karpathy et al. *(FeiFei)* [[pdf]](http://vision.stanford.edu/pdf/karpathy14.pdf) :sparkles:
- DeepPose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy *(Google)* [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
- Two-stream convolutional networks for action recognition in videos (2014), K. Simonyan et al. [[pdf]](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
- A survey on human activity recognition using wearable sensors (2013), O. Lara and M. Labrador [[pdf]](http://romisatriawahono.net/lecture/rm/survey/computer%20vision/Lara%20-%20Human%20Activity%20Recognition%20-%202013.pdf)
- 3D convolutional neural networks for human action recognition (2013), S. Ji et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_JiXYY10.pdf)
- Action recognition with improved trajectories (2013), H. Wang and C. Schmid [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf)
- Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis (2011), Q. Le et al. [[pdf]](http://robotics.stanford.edu/~wzou/cvpr_LeZouYeungNg11.pdf)

### Recurrent Neural Network Models
- Hybrid computing using a neural network with dynamic external memory (2016), A. Graves et al. [[pdf]](https://www.gwern.net/docs/2016-graves.pdf)
- Improved semantic representations from tree-structured long short-term memory networks (2015), K. Tai et al. [[pdf]](https://arxiv.org/pdf/1503.00075)
- End-to-end memory networks (2015), S. Sukbaatar et al. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- Conditional random fields as recurrent neural networks (2015), S. Zheng and S. Jayasumana. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf) :sparkles:
- Memory networks (2014), J. Weston et al. [[pdf]](https://arxiv.org/pdf/1410.3916)
- Neural turing machines (2014), A. Graves et al. [[pdf]](https://arxiv.org/pdf/1410.5401)


### Natural Language Process
- A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation (2016), J. Chung et al. [[pdf]]
- A thorough examination of the cnn/daily mail reading comprehension task (2016), D. Chen et al. [[pdf]](https://arxiv.org/pdf/1606.02858)
- Achieving open vocabulary neural machine translation with hybrid word-character models, M. Luong and C. Manning. [[pdf]](https://arxiv.org/pdf/1604.00788)
- Very Deep Convolutional Networks for Natural Language Processing (2016), A. Conneau et al. [[pdf]](https://arxiv.org/pdf/1606.01781)
- Bag of tricks for efficient text classification (2016), A. Joulin et al. [[pdf]](https://arxiv.org/pdf/1607.01759)
- Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation (2016), Y. Wu et al. *(Le, Vinyals, Dean: Google)* [[pdf]](https://arxiv.org/pdf/1609.08144)
- Exploring the limits of language modeling (2016), R. Jozefowicz et al. *(Vinyals: DeepMind)* [[pdf]](http://arxiv.org/pdf/1602.02410)
- Teaching machines to read and comprehend, K. Hermann et al. [[pdf]](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf)
- Effective approaches to attention-based neural machine translation (2015), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1508.04025)
- A neural conversational model (2015), O. Vinyals and Q. Le. *(Vinyals, Le: Google)* [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)
- Character-aware neural language models (2015), Y. Kim et al. [[pdf]](https://arxiv.org/pdf/1508.06615)
- Grammar as a foreign language (2015), O. Vinyals et al. *(Vinyals, Sutskever, Hinton: Google)* [[pdf]](http://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf)
- Towards AI-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](http://arxiv.org/pdf/1502.05698)
- Neural machine translation by jointly learning to align and translate (2014), D. Bahdanau et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1409.0473) :sparkles:
- Sequence to sequence learning with neural networks (2014), I. Sutskever et al. *(Sutskever, Vinyals, Le: Google)* [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :sparkles:
- Learning phrase representations using RNN encoder-decoder for statistical machine translation (2014), K. Cho et al. *(Bengio)* [[pdf]](http://arxiv.org/pdf/1406.1078)
- A convolutional neural network for modelling sentences (2014), N. Kalchbrenner et al. [[pdf]](http://arxiv.org/pdf/1404.2188v1)
- Convolutional neural networks for sentence classification (2014), Y. Kim [[pdf]](http://arxiv.org/pdf/1408.5882)
- Addressing the rare word problem in neural machine translation (2014), M. Luong et al. [[pdf]](https://arxiv.org/pdf/1410.8206)
- The stanford coreNLP natural language processing toolkit (2014), C. Manning et al. [[pdf]](http://www.surdeanu.info/mihai/papers/acl2014-corenlp.pdf) :sparkles:
- Glove: Global vectors for word representation (2014), J. Pennington et al. [[pdf]](http://anthology.aclweb.org/D/D14/D14-1162.pdf) :sparkles:
- Distributed representations of sentences and documents (2014), Q. Le and T. Mikolov *(Le, Mikolov: Google)* [[pdf]](http://arxiv.org/pdf/1405.4053) *(Google)* :sparkles:
- Distributed representations of words and phrases and their compositionality (2013), T. Mikolov et al. *(Google)* [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) :sparkles:
- Efficient estimation of word representations in vector space (2013), T. Mikolov et al. *(Google)* [[pdf]](http://arxiv.org/pdf/1301.3781) :sparkles:
- Devise: A deep visual-semantic embedding model (2013), A. Frome et al., *(Mikolov: Google)* [[pdf]](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)
- Word representations: a simple and general method for semi-supervised learning (2010), J. Turian *(Bengio)* [[pdf]](http://www.anthology.aclweb.org/P/P10/P10-1040.pdf)

- Recursive deep models for semantic compositionality over a sentiment treebank (2013), R. Socher et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf) :sparkles:
- Linguistic Regularities in Continuous Space Word Representations (2013), T. Mikolov et al. *(Mikolov: Microsoft)* [[pdf]](http://www.aclweb.org/anthology/N13-1#page=784)
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](http://arxiv.org/pdf/1103.0398) :sparkles:
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)

### Reinforcement Learning
- End-to-end training of deep visuomotor policies (2016), S. Levine et al. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/source/15-522.pdf) :sparkles:
- Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection (2016), S. Levine et al. [[pdf]](https://arxiv.org/pdf/1603.02199)
- Continuous deep q-learning with model-based acceleration (2016), S. Gu et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/gu16.pdf)
- Continuous deep q-learning with model-based acceleration (2016), S. Gu et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/gu16.pdf)
- Asynchronous methods for deep reinforcement learning (2016), V. Mnih et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf)
- Mastering the game of Go with deep neural networks and tree search (2016), D. Silver et al. *(Sutskever: DeepMind)* [[pdf]](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) :sparkles:
- Continuous control with deep reinforcement learning (2015), T. Lillicrap et al. [[pdf]](https://arxiv.org/pdf/1509.02971)
- Trust Region Policy Optimization (2015), J. Schulman et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf)
- Human-level control through deep reinforcement learning (2015), V. Mnih et al. *(DeepMind)* [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf) :sparkles:
- Deep learning for detecting robotic grasps (2015), I. Lenz et al. [[pdf]](http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
- Playing atari with deep reinforcement learning (2013), V. Mnih et al. [[pdf]](http://arxiv.org/pdf/1312.5602.pdf))


### Speech / Other Domain
- Automatic speech recognition - A deep learning approach (Book, 2015), D. Yu and L. Deng. [[html]](http://www.springer.com/us/book/9781447157786)
- Deep speech 2: End-to-end speech recognition in English and Mandarin (2015), D. Amodei et al. [[pdf]](https://arxiv.org/pdf/1512.02595) 
- Towards end-to-end speech recognition with recurrent neural networks (2014), A. Graves and N. Jaitly. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)
- Speech recognition with deep recurrent neural networks (2013), A. Graves *(Hinton)* [[pdf]](http://arxiv.org/pdf/1303.5778.pdf)
- Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups (2012), G. Hinton et al. [[pdf]](http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf) :sparkles:
- Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition (2012) G. Dahl et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf) :sparkles:
- Acoustic modeling using deep belief networks (2012), A. Mohamed et al. *(Hinton)* [[pdf]](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)

* * *

### New papers
*Newly released papers (< 6 months) which are worth reading*
- Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models, S. Ioffe. *(Google)* [[pdf]](https://arxiv.org/abs/1702.03275)
- Understanding deep learning requires rethinking generalization, C. Zhang et al. *(Vinyals)* [[pdf]](https://arxiv.org/pdf/1611.03530)
- WaveNet: A Generative Model for Raw Audio (2016), A. Oord et al. *(DeepMind)* [[pdf]](https://arxiv.org/pdf/1609.03499v2) [[web]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- Layer Normalization (2016), J. Ba et al. *(Hinton)* [[pdf]](https://arxiv.org/pdf/1607.06450v1.pdf)
- Deep neural network architectures for deep reinforcement learning, Z. Wang et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1511.06581.pdf)
- Learning to learn by gradient descent by gradient descent (2016), M. Andrychowicz et al. *(DeepMind)* [[pdf]](http://arxiv.org/pdf/1606.04474v1)
- Adversarially learned inference (2016), V. Dumoulin et al. [[web]](https://ishmaelbelghazi.github.io/ALI/)[[pdf]](https://arxiv.org/pdf/1606.00704v1)
- Understanding convolutional neural networks (2016), J. Koushik [[pdf]](https://arxiv.org/pdf/1605.09081v1)
- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size (2016), F. Iandola et al. [[pdf]](http://arxiv.org/pdf/1602.07360)
- Learning to compose neural networks for question answering (2016), J. Andreas et al. [[pdf]](http://arxiv.org/pdf/1601.01705)
- Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection (2016) *(Google)*, S. Levine et al. [[pdf]](http://arxiv.org/pdf/1603.02199v3)
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Eie: Efficient inference engine on compressed deep neural network (2016), S. Han et al. [[pdf]](http://arxiv.org/pdf/1602.01528)
- Adaptive Computation Time for Recurrent Neural Networks (2016), A. Graves [[pdf]](http://arxiv.org/pdf/1603.08983)
- Densely connected convolutional networks (2016), G. Huang et al. [[pdf]](https://arxiv.org/pdf/1608.06993v1)

### Old Papers
*Classic papers (1997~2011) which cause the advent of deep learning era*
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)
- Learning deep architectures for AI (2009), Y. Bengio. [[pdf]](http://sanghv.com/download/soft/machine%20learning,%20artificial%20intelligence,%20mathematics%20ebooks/ML/learning%20deep%20architectures%20for%20AI%20(2009).pdf)
- Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations (2009), H. Lee et al. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf)
- Greedy layer-wise training of deep networks (2007), Y. Bengio et al. [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- Reducing the dimensionality of data with neural networks, G. Hinton and R. Salakhutdinov. [[pdf]](http://homes.mpimf-heidelberg.mpg.de/~mhelmsta/pdf/2006%20Hinton%20Salakhudtkinov%20Science.pdf)
- A fast learning algorithm for deep belief nets (2006), G. Hinton et al. [[pdf]](http://nuyoo.utm.mx/~jjf/rna/A8%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)
- Gradient-based learning applied to document recognition (1998), Y. LeCun et al. [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- Long short-term memory (1997), S. Hochreiter and J. Schmidhuber. [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735)

### More than Top 100
- Gated Feedback Recurrent Neural Networks (2015), J. Chung et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/chung15.pdf)
- Pointer networks (2015), O. Vinyals et al. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf)
- Visualizing and Understanding Recurrent Networks (2015), A. Karpathy et al. [[pdf]](https://arxiv.org/pdf/1506.02078)
- Attention-based models for speech recognition (2015), J. Chorowski et al. [[pdf]](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)

### HW / SW / Dataset
- OpenAI gym (2016), G. Brockman et al. [[pdf]](https://arxiv.org/pdf/1606.01540)
- TensorFlow: Large-scale machine learning on heterogeneous distributed systems (2016), M. Abadi et al. [[pdf]](http://arxiv.org/pdf/1603.04467) :sparkles:
- Theano: A Python framework for fast computation of mathematical expressions, R. Al-Rfou et al. *(Bengio)*
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](http://arxiv.org/pdf/1408.5093) :sparkles:

### Book / Survey / Review
- Deep learning (Book, 2016), Goodfellow et al. [[html]](http://www.deeplearningbook.org/)
- LSTM: A search space odyssey (2016), K. Greff et al. [[pdf]](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[pdf]](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) :sparkles:
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](http://arxiv.org/pdf/1404.7828) :sparkles:

- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](http://arxiv.org/pdf/1206.5538) :sparkles:

### Video Lectures / Tutorials / Blogs


### Distinguished Researchers
*Distinguished deep learning researchers who have published +3 (:sparkles: +6) papers on the awesome list*
 (The papers in *Hardware / Software*, *Papers Worth Reading*, *Classic Papers* sections are excluded in counting.)

- [Chirstian Szegedy](https://scholar.google.ca/citations?hl=en&user=3QeF7mAAAAAJ), *Google* :sparkles:
- [Kaiming He](https://scholar.google.ca/citations?user=DhtAFkwAAAAJ), *Facebook* :sparkles:
- [Geoffrey Hinton](https://scholar.google.ca/citations?user=JicYPdAAAAAJ), *Google* :sparkles:
- [Ilya Sutskever](https://scholar.google.ca/citations?user=x04W_mMAAAAJ), *OpenAI* :sparkles:
- [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ), *OpenAI* :sparkles:
- [Oriol Vinyals](https://scholar.google.ca/citations?user=NkzyCvUAAAAJ), *Google DeepMind* :sparkles:
- [Quoc Le](https://scholar.google.ca/citations?user=vfT6-XIAAAAJ), *Google* :sparkles:
- [Tomas Mikolov](https://scholar.google.ca/citations?hl=en&user=oBu8kMMAAAAJ), *Facebook*
- [Yann LeCun](https://scholar.google.ca/citations?user=WLN3QrAAAAAJ), *Facebook* :sparkles:
- [Yoshua Bengio](https://scholar.google.ca/citations?user=kukA0LcAAAAJ), *University of Montreal* :sparkles:

- [Aaron Courville](https://scholar.google.ca/citations?user=km6CP8cAAAAJ), *University of Montreal*
- [Alex Graves](https://scholar.google.ca/citations?user=DaFHynwAAAAJ), *Google DeepMind*
- [Andrej Karpathy](https://scholar.google.ca/citations?hl=en&user=l8WuQJgAAAAJ), *OpenAI*
- [Andrew Ng](https://scholar.google.ca/citations?user=JgDKULMAAAAJ), *Baidu*
- [Andrew Zisserman](https://scholar.google.ca/citations?user=UZ5wscMAAAAJ), *University of Oxford*

- [Christopher Manning](https://scholar.google.ca/citations?hl=en&user=1zmDOdwAAAAJ), *Stanford University*
- [David Silver](https://scholar.google.ca/citations?user=-8DNE4UAAAAJ), *Google DeepMind*
- [Dong Yu](https://scholar.google.ca/citations?hl=en&user=tMY31_gAAAAJ), *Microsoft Research*
- [Ross Girshick](https://scholar.google.ca/citations?user=W8VIEZgAAAAJ), *Facebook*

- [Karen Simonyan](https://scholar.google.ca/citations?user=L7lMQkQAAAAJ), *Google DeepMind*
- [Kyunghyun Cho](https://scholar.google.ca/citations?user=0RAmmIAAAAAJ), *New York University*
- [Honglak Lee](https://scholar.google.ca/citations?hl=en&user=fmSHtE8AAAAJ), *University of Michigan*

- [Jeff Dean](https://scholar.google.ca/citations?user=NMS69lQAAAAJ), *Google*,
- [Jeff Donahue](https://scholar.google.ca/citations?hl=en&user=UfbuDH8AAAAJ), *U.C. Berkeley*
- [Jian Sun](https://scholar.google.ca/citations?user=ALVSZAYAAAAJ), *Microsoft Research*
- [Juergen Schmidhuber](https://scholar.google.ca/citations?user=gLnCTgIAAAAJ), *Swiss AI Lab IDSIA*
- [Li Fei-Fei](https://scholar.google.ca/citations?hl=en&user=rDfyQnIAAAAJ), *Stanford University*

- [Pascal Vincent](https://scholar.google.ca/citations?user=WBCKQMsAAAAJ), *University of Montreal*
- [Rob Fergus](https://scholar.google.ca/citations?user=GgQ9GEkAAAAJ), *Facebook, New York University*
- [Ruslan Salakhutdinov](https://scholar.google.ca/citations?user=ITZ1e7MAAAAJ), *CMU*
- [Trevor Darrell](https://scholar.google.ca/citations?user=bh-uRFMAAAAJ), *U.C. Berkeley*

## Acknowledgement

Thank you for all your contributions. Please make sure to read the [contributing guide](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md) before you make a pull request.

You can follow my [facebook page](https://www.facebook.com/terryum.io/), [twitter](https://twitter.com/TerryUm_ML) or [google plus](https://plus.google.com/+TerryTaeWoongUm/) to get useful information about machine learning and deep learning.

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Terry T. Um](https://www.facebook.com/terryum.io/) has waived all copyright and related or neighboring rights to this work.
