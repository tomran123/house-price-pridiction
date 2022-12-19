# house-price-pridiction
Hi, everyone!
这里我第一个github项目，在这个项目里，我要完成一个kaggle上面的竞赛题目，这个竞赛是关于预测房价的，在这个题目里，我们要根据房子的79个特征，如质量，面积，街区，壁炉等，预测相应的房价，数据集分为test.csv和train.csv。


1.首先导入所需要的库：
![image](https://user-images.githubusercontent.com/110798232/208420988-1a538ce8-62fe-4d29-9343-bb857a121a18.png)


2.读取训练集和测试集，检查数据大小，查看前5行的数据：
![image](https://user-images.githubusercontent.com/110798232/208421535-a485cf15-d3f5-4797-8984-909124435749.png)

![image](https://user-images.githubusercontent.com/110798232/208421341-ff2e88c3-130f-4754-a59f-087b4f6b3271.png)

![image](https://user-images.githubusercontent.com/110798232/208421804-77ec32c4-2045-41ea-86ee-003b50ed2537.png)

![image](https://user-images.githubusercontent.com/110798232/208421834-17d20295-c64a-4484-826c-9747da6a0803.png)

训练集比测试集多出来一列加格列，也是我们测试集的预测目标

![image](https://user-images.githubusercontent.com/110798232/208422173-aeb978e2-b08a-49ea-9172-7b5c9f691ac2.png)
删除第一行id


3.用matplotlib绘制房屋价格
![image](https://user-images.githubusercontent.com/110798232/208422873-7d1e6b93-6e68-44ec-9854-5cda4348b318.png)

![image](https://user-images.githubusercontent.com/110798232/208423021-87fb7c3a-aa0d-4b0c-b4a7-03f185e0f8b0.png)

通过图像，我们发现是右偏分布，和正态分布有差距，我们再看一下这个分布曲线的峰度和偏度
