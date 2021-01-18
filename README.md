# LR_CNN

## 项目文件说明
-- baseline  
  --- layers.py 定义了卷积层，池化层，全连接层的前馈与基于LR的反向传播函数，BP估计梯度后期更新;SGD，Adam优化器等  
  --- model.py 定义了Lenet-5，一层卷积+两层全连接，两层全连接的模型结构  
  --- setting.py 定义了代码运行环境设置，目前仅为设置设备号即GPU device  
  --- train.py 模型训练代码，需要手动修改的变量为batch_size 批数量，learning_rate 学习率，repeat_n 重复样本的数量（p.s. 重复样本是沿着0轴重复的）  
  --- utils.py 定义了im2col,col2im的实现(将卷积转为矩阵乘法运算)  
  以上所有功能均基于pytorch实现  
  
  ## 当前实现情况  
  目前仍然只能训练下全连接网络，卷积神经网络无法训练（始终处于瞎猜的水平即10%）    
  卷积层对于权重和噪声的scaling系数的梯度均已实现,但仍待检查  
  p.s. 梯度的计算大量依赖于einsum函数，利用下标对应关系计算
