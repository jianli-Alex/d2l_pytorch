## d2l_pytorch
动手学深度学习Pytorch版的个人笔记

1. Pytorch的基础用法
	- 数据操作/各种常用函数/自动梯度
	- Pytorch和numpy操作之间的比较/结果对比
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
	- pytorch版的mlp从零实现和简单实现
	- 实现各种softmax/tanh/relu等激活函数
	- 为通用训练函数(train_pytorch)添加画图功能，测试各种情况
5. optimization
	- 正常拟合/过拟合/欠拟合实验(以多项式函数拟合来做实验)
	- 权重衰减在高维回归中的实验(包括只更新权重的权重衰减)
	- 对权重衰减是否除以batch_size的对比实验见"2. linear model"中的"linear_bridge"相关实验
	- 优化器自定义参数权重衰减率时不更新实验
	- dropout的两种方式实现/为网络自定义添加dropout和pytorch实验
	- GPU实验/分割通用训练函数为(experiment和pytorch版)，增加GPU的相关支持
