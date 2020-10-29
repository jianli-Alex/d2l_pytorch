## d2l_pytorch
动手学深度学习Pytorch版的个人实现

1. Pytorch的基础用法
	- 数据操作/各种常用函数/自动梯度
	- Pytorch和numpy操作之间的比较/结果对比
	- Pytorh的nn.Module/初始化/自定义/cuda
2. Linear model
	- 实现linear model的numpy版/进度条的实现（见sqdm）/小批量划分数据集等通用函数
	- 实现linear model的正规方程解法
	- linear model的Pytorch从零实现和简洁实现
	- linear model的L2正则化（大多使用L2来实现权重衰减）实现（numpy/pytorch版），对比pytorch的权重衰减方法和我们熟知的L2实现方法的异同
3. logistic and softmax model
	- 实现logistic and softmax的numpy版
	- 实现自助法/进度条实现改进/加载fashion_mnist等函数
	- pytorch版softmax的从零实现和简洁实现
	- 实现通用的pytorch训练函数
4. perceptron and multiple perceptron
	- 实现感知机的numpy版/one vs rest/one vs one策略的多分类情况
	- 使用numpy手动实现反馈网络和多层感知机
	- pytorch版的mlp从零实现和简单实现
	- 实现各种softmax/tanh/relu等激活函数
5. optimization
	- 正常拟合/过拟合/欠拟合实验(以多项式函数拟合来做实验)
	- 权重衰减在高维回归中的实验(包括只更新权重的权重衰减)
	- 对权重衰减是否除以batch_size的对比实验见"2. linear model"中的"linear_bridge"相关实验
	- 优化器自定义参数权重衰减率时不更新实验
	- dropout的两种方式实现/为网络自定义添加dropout和pytorch实验
6. CNNs
	- 卷积实验（自定义实现以及利用nn.Module实现）
	- 实现LeNet5/AlexNet/VGG11/VGG16/NIN/GoogleNet
	- 实验局部响应归一化(目前基本不用)
	- 自定义实现BN和Pytorch简洁实现
7. RNNs
	- RNN的相关实验
8. d2_func
	- 通用训练函数(train_experiment/train_pytorch/train_epoch)，train_experiment用于个人的实验（包含非pytorch实现如手动实现优化器），train_pytorch只包含pytorch的实现（兼容iteration和epoch的绘图，但是测试的时候每一iteration都会测试，如果传进了需要测试的参数），train_epoch只用于epoch的训练（绘图时只绘制epoch，因此1个epoch只测试一次，训练速度加快）
	- train_pytorch和tran_epoch都增加了梯度累加的支持，GPU支持，train_experiment没有添加，这种训练模式支持绘图（train_pytorch和train_experiment支持iteration/epoch绘图，且train_pytorch支持抽样绘图，train_epoch仅支持epoch绘图，但可绘制训练集的每个batch在epoch的平均loss/score或者最后一个iteraction的loss/score）
	- 数据加载与划分函数(data_prepare.py)/绘图设置函数(draw.py)/自实现优化器和损失函数(optim.py)/模型选择（自助法）
	- 进度条实现(sqdm)，用于监测训练过程中的loss和score的变化，已经集成到model_train.py的各种训练模式中
	- 代码比较函数(compare_file)，用于比较两个py文件的代码异同（以网页的方式）
	- 目录树的自实现（tree.py），可用于以目录树的形式打印任意目录下的文件
