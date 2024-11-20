# yjsprj1   
## 一、游戏说明   
画线连点，一共20关，玩家根据点的分布连线，每关保存1个csv文件和一张图片   
## 二、实验数据解说   
1.**csv数据**   
csv压缩文件解压后有20个文件夹，存储着20关游戏若干玩家的csv数据，csv数据包含玩家编号、关卡号、第几笔、光标位置及时刻   
2.**png数据**   
20关对应的结果图片   
## 三、实验步骤   
### （一）分析数据   
考虑到同一关卡不同玩家每笔画线的顺序可能不同，所画笔数不同，画线时长不同每一笔记录的光标数量不同，需要选取可量化的特征来代表每关的整体特性，最终选择每关路径总长度和平均曲率。   
### （二）数据处理   
1.提取关卡特征（路径总长度和平均曲率）   
遍历每个关卡文件夹下所有玩家的CSV数据，每关路径总长度=该关卡所有玩家路径长度总和/玩家数量，每关平均曲率=该关卡所有玩家平均曲率总和/玩家数量，这里曲率简化作简化处理   
2.对特征归一化处理   
### （三）计算差异度矩阵绘制热力图   
### （四）层次聚类绘制并结果可视化   
### （五）t-SNE降维   
