# house-price-pridiction
Hi, everyone!
这里我第一个github项目，在这个项目里，我要完成一个kaggle上面的竞赛题目，这个竞赛是关于预测房价的，在这个题目里，我们要根据房子的79个特征，如质量，面积，街区，壁炉等，预测相应的房价，数据集分为test.csv和train.csv。这个测试集和训练集数据以及代码我放到后面的文件里了。


在这个项目中，我们先用matplotlib对房屋的价格进行了分析，发现是明显的右偏分布，偏度较大，所以后期我们要进行一些处理，使其像正态分布靠拢，所以我们通过对数变换，调到标准的正态分布。
接着，进行异常值和缺失值的处理，对于缺少的部分离散型数据，我们用众数填充缺失值，
Functional居家功能性，数据描述说NA表示类型"Typ'。因此，我们用其填充。
LotFrontage，由于房屋到街道的距离，最有可能与其附近其他房屋到街道的距离相同或相似，因此我们可以通过该社区的LotFrontage中位数来填充缺失值：
最后我们删除Utlities这个特征，这个特征除了一个-NoSeWNa'和2个 NA，其余值都是 AlIPub'，因此该项特征的方差非常小。这个特征对预测建模没有帮助。因此，我们可以安全地删除它。
然后我们再把部分数值特征转化为类别特征，对高偏度的数值型特征进行Box——Cox转换。创建和目标特征相关的新的特征。
通过这一系列处理后，训练集列数增加到163，测试集增加到162


分析并处理完数据后，我们要进行模型的预测了，我采用了三种模型预测的方法：岭回归，lasso和随机森林，通过比较，最终选择错误率最低的lasso模型


