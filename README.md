## 2020百度之星开发者大赛：交通标识检测与场景匹配 - 解决方案
****************************************************************************************************************************************
- **【赛事信息】** [2020百度之星开发者大赛：交通标识检测与场景匹配](https://aistudio.baidu.com/aistudio/competition/detail/39)
- **【参赛队伍】** 请注意交通标志    
- **【参赛名次】** 复赛1/50     初赛1/2607     
- **【团队成员】** 顾竟潇[@seigato](https://github.com/gujingxiao)(Calmcar)、邓毓弸[@BLING-1994](https://github.com/BLING-1994)(中科院空天院)

****************************************************************************************************************************************
### 模型文件
- **【模型说明】** 本次比赛所有模型和配置文件均已上传百度云盘      
- **【提取路径】** https://pan.baidu.com/s/1n3fUtIBmrbOwjjgRgtu6lQ  
- **【提取码】** od8i      

****************************************************************************************************************************************
### 系统环境
- **python 3.6.5**     
- **paddlepaddle-gpu 1.8.0-post107**   
- **opencv-python 3.3.0.10**      
- **opencv-contrib-python 3.3.0.10**   
- **numpy 1.17.5**   
****************************************************************************************************************************************
### 方案介绍
  **【1】** 本次比赛涉及的技术方面很广泛，在Baseline的框架基础上，针对比赛中不同阶段，我们使用了PaddleDetection、PaddleClas、PaddleMetricLearning和自行设计的star2020的后处理部分    
    
  **【2】** PaddleDetection主要用于检测交通标识，本次比赛大类3类，小类共19类，我们分别进行了3大类检测训练，并进行了单模型检测和多模型融合检测  
    
  **【3】** PaddleClas主要用于训练19类细分类模型，对检测结果进行细分类和去误检；并且分类模型将会作为匹配训练的预训练模型使用  
    
  **【4】** PaddleMetricLearning主要用于训练匹配模型，输出特征向量，和每一个匹配对的cosine distance
    
  **【5】** star2020中主要包含标签、数据可视化、融合脚本、后处理脚本、本地评估脚本、结果分析脚本等内容，通过这些脚本可以在验证集上进行验证，作为分数提升的依据     
    
  **【6】** 最后，参照Star2020得到的所有内容的评估结果，对test进行处理，提交成绩        
  
****************************************************************************************************************************************

### 详细处理流程 
  - **数据准备**        
  
    【1】 将traffic数据集存放在某一个路径下，然后建立train和test两个目录，分别存放训练集和测试集     

    【2】 在train目录下，放入对应的input、pic、tag

    【3】 在test目录下，放入对应的input、pic

    【4】 使用star2020/process_tag/splitValTag.py脚本，进行训练集和验证集的随机划分，这里由于数据集较大的原因，我们没有使用5-folds，只是随机生成了一组；生成好的tag存放在train/train_tag和train/val_tag下  
    
    【5】 使用star2020/process_classification/convertC19toC3.py脚本，将val_tag转换为val_tag3，也存放在train目录下
    
****************************************************************************************************************************************

  - **检测训练与验证**  
  
    【1】 根据PaddleDetection中ModelZoo里模型能力的描述，我们选择了其中更为接近SOTA效果的模型，并新增了一部分模型，共计9个模型，所有configs文件均存放在PaddleDetection/configs/star2020/中。分别为：
    
     |Models|Threshold|MAP|F1|
     |:---|:---|:---|:---|
     |cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal|0.65|0.8669|0.8771|
     |cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal|0.65|0.8730|0.8843|
     |cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal|0.65|0.8701|0.8578|
     |cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal|0.65|0.8770|0.8682|
     |cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal|0.65|0.8790|0.8715|
     |cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal|0.65|0.8701|0.8668|
     |cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal|0.65|0.8804|0.8803|
     |cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn|0.65|0.8723|0.8629|
     |cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas|0.65|0.8721|0.8619|      

    【2】 参照PaddleDetection训练方式，将PaddleDetection/scripts/train.sh中相应的内容，修改为所需要的训练网络配置文件，并修改相关配置文件，例如metric、dataset等           
    
    【3】 考虑到本次比赛类别不均衡的问题，我们在检测训练时只分了3类训练，所以class num为4。这里我们注册了一个trafficGeneralDataset的类来进行19类到3类的映射关系和reader     
    
    【4】 修改完成后，运行训练脚本： PaddleDetection/scripts/train.sh，这里不建议在训练过程中进行验证（PaddleDetection存在bug，cascade模型训练过程中验证可能报错）        
    
    【5】 验证： PaddleDetection/scripts/eval.sh 其中当输入save_prediction_only=true时，将会直接生成检测结果的文件；建议验证时改为单卡，并且在reader里面将worker_num改为1(防止报错)     
    
    【6】 生成的检测结果可以使用star2020/detectEval_F1.py进行验证，将会输出不同置信度下的precision、recall和F1-score的信息(可见star2020中Result_analysis文件)      
    
****************************************************************************************************************************************    

  - **细分类训练、误检分类训练及验证**  

    【1】 首先进行数据提取和数据集制作，这一步骤仿照imagenet的数据存储方式，数据集制作脚本star2020/process_classification/    
    
    【2】 细分类训练使用star2020/process_classification/getTrafficImages.py将所有标注好的图像中的目标分类别存储 
    
    【3】 误检分类使用genTrueDetectImage.py和getFalseDetectImages.py进行生成       
    
    【4】 使用star2020/process_classification/genClasTrainValList.py对已经提取好并分好类的数据进行训练集和验证集的区分，同时生成train_list.txt和val_list.txt（此步骤针对细分类和误检分类训练分别进行）      
    
    【5】 然后使用PaddleClas/tools/train.py脚本进行训练和验证      
    
    【6】 训练完成后，使用PaddleClas/traffic_class_infer.py脚本进行检测结果的重分类，并输出验证后的完整结果      
    
    【7】 生成的结果可以使用star2020/ process_detection/detectEval_F1.py进行验证 

****************************************************************************************************************************************    

  - **检测模型融合及误检筛选**  
  
    【1】 将所有的单检测模型训练好后，生成检测结果，然后使用细分类模型重新校验检测结果，生成每一个检测模型的最终检测结果     
    
    【2】 使用star2020/process_detection/detectEnsemble.py脚本进行检测模型结果的融合处理，将所有生成好的结果，生成融合的检测结果        
    
    【3】 然后使用star2020/ process_detection/detectEval_F1.py进行验证，找到阈值最好或者阈值较好的结果        
    
    【4】 使用star2020/ process_detection/getDetectThres.py脚本进行阈值选择，将满足阈值的结果重新存放       
    
    【5】 使用PaddleClas/traffic_class_infer_filter_false.py脚本进行检测融合结果的误检筛选，并输出筛选后的完整结果       
    
****************************************************************************************************************************************        

  - **匹配模型训练与验证**  
  
    【1】 根据PaddleClas中ModelZoo里模型能力的描述，我们认为分类更强的模型其特征提取能力更强，结合算力评估，我们选择了其中性价比高且更为接近SOTA效果的模型，共计5个模型。分别为：
    
     |Models|Threshold|F1|
     |:---|:---|:---|
     |ResNeXt101_vd_64x4d|0.65|0.8539|
     |ResNeXt152_vd_64x4d|0.65|0.8546|
     |ResNeXt152_vd_32x4d|0.65|0.8541|
     |SE-ResNeXt50_vd_32x4d|0.65|0.8460|
     |ResNet50_vd|0.65|0.8513|

    【2】 匹配模型训练流程参考了官方Baseline的脚本，但是略有修改(删除了reader中获取每一个roi时的数据增强，因为这种增强将会导致难收敛、并且经测试会导致cosine distance分布不均匀)，并提高分辨率至160*160，在models中新增了多个模型结构文件以适应多模型的训练   
    
    【3】 训练完成后，使用metric_learning_traffic/eval.sh脚本，对上一步选好的检测结果进行匹配，并生成匹配结果     
    
    【4】 匹配结果可以通过使用star2020/ process_match/matchsEval_F1.py进行验证。在val_tag的基础上直接使用匹配模型，评估结果    
    
****************************************************************************************************************************************            
 
  - **匹配模型融合与后处理**  
  
    【1】 使用star2020/process_match/pair_ensemble.py脚本可以进行融合处理，即加权平均       
    
    【2】 使用star2020/process_match/processMatchTool/npytolist.py脚本生成序列内匹配特征文件      
    
    【3】 使用star2020/process_match/processMatchTool/ postProcessMatchs.py脚本进行匹配后处理，涉及到的处理逻辑和功能在cascadeFilter.py脚本中，详细功能如下：  
    
    |函数|功能|
    |:---|:---|
    |correctWrongOrder|将顺序不对或者异常的进行删除或重排序|
    |deleteWrongTypeMatch|将两个目标类型不同的进行删除|
    |filterThresholdMatch|通过检测阈值来过滤检测目标和匹配目标|
    |deleteUnreasonableMatch|根据位置信息删除一些毫无依据的匹配目标|
    |filterMultiToOneMatch|针对同一张图不同目标匹配同一个的情况，进行过滤处理|
    |filterOneToMultiMatch|针对一个匹配上同一张图多个目标的情况，进行过滤处理|
    |getLostMatchBack|通过单一序列内的相似度匹配，找到同一目标，然后针对可靠的匹配进行互补，将丢失的匹配对补回来|
    |getNonMatchBack|将所有没有匹配上的目标找回，并写入空匹配|
    
    【4】 处理结果可以通过使用star2020/ process_match/matchsEval_F1.py进行验证集分数验证    
    
    【5】 如使用测试集，后处理完成后，就可以提交到测试榜中等待评分了    

****************************************************************************************************************************************          
    
### 复赛成绩记录
- **【单检测模型Cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal + 细分类ResNeXt101_vd_64x4d + 匹配模型SE-ResNeXt50_vd_32x4d】**  
  
  |Models|Detect Threshold|Match Threshold|Val F1 Score|Test F1 Score|
  |:---|:---|:---|:---|:---|
  |Single|0.70|0.35|0.70341|0.67505|


- **【8x检测模型融合 + 细分类ResNeXt101_vd_64x4d + 5x匹配模型融合】** 
  |Models|Detect Threshold|Match Threshold|Val F1 Score|Test F1 Score|
  |:---|:---|:---|:---|:---|
  |Ensemble|0.65|0.20|0.7677|0.72994|

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
