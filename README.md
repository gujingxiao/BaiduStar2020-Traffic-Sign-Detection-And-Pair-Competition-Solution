## 2020百度之星开发者大赛：交通标识检测与场景匹配 - 解决方案
****************************************************************************************************************************************
- **【赛事信息】** [2020百度之星开发者大赛：交通标识检测与场景匹配](https://aistudio.baidu.com/aistudio/competition/detail/39)
- **【参赛队伍】** 请注意交通标志    
- **【参赛名次】** 初赛 1/2231    
- **【团队成员】** 顾竟潇(Calmcar)、邓毓弸(中科院)

****************************************************************************************************************************************
### 模型文件
  **【模型说明】** 本次比赛所有模型和配置文件均已上传百度云盘    
    
  **【提取路径】** https://pan.baidu.com/s/1n3fUtIBmrbOwjjgRgtu6lQ  
    
  **【提取码】** od8i      

****************************************************************************************************************************************
### 方案介绍
  **【1】** 本次比赛涉及的技术方面很广泛，在Baseline的框架基础上，针对比赛中不同阶段，我们使用了PaddleDetection、PaddleClas、PaddleMetricLearning和自行设计的star2020的后处理部分    
    
  **【2】** PaddleDetection主要用于检测交通标识，本次比赛大类3类，小类共19类，我们分别进行了模型的训练，并进行了单模型检测和多模型融合检测  
    
  **【3】** PaddleClas主要用于检验检测出来的类别是否正确，训练模型进行19类再分类，对检测结果进行校验和修正；并且分类模型将会作为匹配训练的预训练模型使用  
    
  **【4】** PaddleMetricLearning主要用于训练匹配模型，输出特征向量，和每一个匹配对的cosine distance
    
  **【5】** star2020中主要包含检测模型融合、匹配模型融合、验证集评估、匹配后处理等内容，通过这些脚本可以得到可靠的结果，并大幅度提升分数      
    
  **【6】** 最后，参照Star2020得到的所有内容的评估结果，对test进行处理，提交成绩        
  
****************************************************************************************************************************************

### 详细处理流程 
  - **数据准备**        
  
    【1】 将traffic数据集存放在某一个路径下，然后建立train和test两个目录，分别存放训练集和测试集     

    【2】 在train目录下，放入对应的input、pic、tag

    【3】 在test目录下，放入对应的input、pic

    【4】 使用star2020/splitValTag.py脚本，进行训练集和验证集的随机划分，这里由于数据集较大的原因，我们没有使用5-folds，只是随机生成了一组；生成好的tag存放在train/train_tag和train/val_tag下  
    
****************************************************************************************************************************************

  - **检测训练与验证**  
  
    【1】 为了防止混淆，本次比赛中使用的所有configs文件都直接存放在了output中，加载时也直接使用output下每一个网络的configs       
    
    【2】 参照PaddleDetection训练方式，将scripts/train.sh中相应的内容，修改为所需要的训练网络配置文件，并修改相关配置文件，例如metric、dataset等       
    
    【3】 考虑到本次比赛类别不均衡的问题，我们在检测训练时只分了3类训练，所以class num为4。这里我们注册了一个trafficGeneralDataset的类来进行19类到3类的映射关系和reader     
    
    【4】 修改完成后，运行训练脚本： ./scripts/train.sh，这里不建议在训练过程中进行验证        
    
    【5】 验证： ./scripts/eval.sh 其中当输入save_prediction_only=true时，将会直接生成检测结果的文件；建议验证时改为单卡，并且在reader里面将worker_num改为1(防止报错)     
    
    【6】 生成的检测结果可以使用star2020/detectEval_F1.py进行验证，将会输出不同置信度下的precision、recall和F1-score的信息(可见star2020中Result_analysis文件)      
    
****************************************************************************************************************************************    

  - **细分类训练与验证**  

    【1】 首先进行数据提取和数据集制作，这一步骤仿照imagenet的数据存储方式，提取和数据集制作脚本存放在star2020/traffic_clas中    
    
    【2】 使用getTrafficImages.py将所有标注好的图像中的目标分类别存储 
    
    【3】 使用genClasTrainValList.py对已经提取好并分好类的数据进行训练集和验证集的区分，同时生成train_list.txt和val_list.txt       
    
    【4】 然后使用PaddleClas/tools/train.py脚本进行训练和验证      
    
    【5】 训练完成后，可使用PaddleClas/traffic_class_infer.py脚本进行检测结果的重验证，并输出验证后的完整结果      
    
    【6】 生成的结果可以使用star2020/detectEval_F1.py进行验证      
    
****************************************************************************************************************************************    

  - **检测模型融合**  
  
    【1】 将所有的单检测模型训练好后，生成检测结果，然后使用细分类模型重新校验检测结果，生成每一个检测模型的最终检测结果     
    
    【2】 使用star2020/detectEnsemble.py脚本进行检测模型结果的融合处理，将所有生成好的结果，输入到融合算法中，生成融合的检测结果        
    
    【3】 然后使用star2020/detectEval_F1.py进行验证，找到阈值最好或者阈值较好的结果       
    
    【4】 使用star2020/getDetectThres.py脚本进行阈值选择，将满足阈值的结果重新存放       
    
****************************************************************************************************************************************        

  - **匹配模型训练与验证**  
  
    【1】 匹配模型训练流程参考了官方Baseline的脚本，但是略有修改(删除了reader中获取每一个roi时的数据增强，因为这种增强将会导致难收敛、并且经测试会导致cosine distance分布不均匀)       
    
    【2】 训练完成后，使用metric_learning_traffic/eval.sh脚本，对上一步选好的检测结果进行匹配，并生成匹配结果   
    
    【3】 匹配结果可以通过使用star2020/matchsEval_F1.py进行验证     
    
****************************************************************************************************************************************            
 
  - **匹配模型融合与后处理**  
  
    【1】 本次比赛一共训练了3个匹配模型，针对每一个匹配对都会生成3个cosine distance，使用star2020/pair_ensemble.py脚本可以进行融合处理，即加权平均       
    
    【2】 将匹配融合的结果，使用star2020/postProcessMatchs.py进行匹配顺序检查、空间位置检查、匹配重复检查、图像相似度检测等多项后处理，并生成最终处理后的提交结果      
    
    【3】 处理结果可以通过使用star2020/matchsEval_F1.py进行验证集分数验证        
    
    【4】 如使用测试集，第【2】步完成后，就可以提交到测试榜中等待评分了     

****************************************************************************************************************************************          
    
### 初赛成绩记录
- **【单检测模型Cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal + 细分类ResNeXt101_vd_64x4d + 匹配模型SE-ResNeXt50_vd_32x4d】**  
  
  |Models|Detect Threshold|Match Threshold|Val F1 Score|Test F1 Score|
  |:---|:---|:---|:---|:---|
  |Single|0.70|0.65|0.6701|0.63167|


- **【6x检测模型融合 + 细分类ResNeXt101_vd_64x4d + 3x匹配模型融合】** 
  |Models|Detect Threshold|Match Threshold|Val F1 Score|Test F1 Score|
  |:---|:---|:---|:---|:---|
  |Ensemble|0.70|0.25|0.7108|0.67132|

### 引用及参考
- PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)      

- PaddleClas (https://github.com/PaddlePaddle/PaddleClas)        

- PaddleModels (https://github.com/PaddlePaddle/models)      

- PaddleResearchLandmark (https://github.com/PaddlePaddle/Research/tree/master/CV/landmark) (https://arxiv.org/pdf/1906.03990.pdf)     

- PaddleMetricLearning (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)       

- Image Matching Challenge 2020 (https://vision.uvic.ca/image-matching-challenge/)      

- AdaLAM (https://arxiv.org/abs/2006.04250)


### 联系方式
This repo is currently maintained by Jingxiao Gu ([@seigato](https://github.com/gujingxiao)), Yupeng Deng ([@BLING-1994](https://github.com/BLING-1994)).
