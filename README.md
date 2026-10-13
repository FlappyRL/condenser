# Condenser
这个markdown大部分都是废话,   
目前只是探索性质的小项目  
个人留个纪念

大概是我一开始不知道有可以给文本生成embedding的模型  
然后从另一个角度思考怎么给文本生成embedding  

可以说EXP_RESIDUAL其实是对transformer架构很有趣的一个小实验  

回头蒸馏自己CoT时候可以用到hh

> 运行以下程序前先CD过去!!!!  
> ./condenser不需要CD

# EXP_RESIDUAL
首先思考一下这个问题  
显然普通的Word2Vec不适用于不定长，没见过的长token串  

所以需要通过attention机制或者某种办法让模型理解 然后再生成"潜在表征" 所以我便想到了之前网上看到的:
- anthropic的那个J-SPACE (用雅可比矩阵处理residual stream. 得到模型内部概念对应的激活?) 
- 以及 abliterated模型 (也是通过观察residual stream. 得到拒绝向量来越狱)

可见residual stream实际上是可以被单独拿出来观测的
于是就让ai写了这么个程序, 首先使用qwen3的basemodel,  
0.6B因为我的电脑很垃圾 `Intel Arc 140T + 16G共享显存`

然后输入五个句子, 对于每个句子, 每一个transformer block. 都保存其residual stream上抽出来的的向量
分为两种:
- `update`: 那一步净更新的
- `residual`: 正常的residual stream向量

然后还有三种pooling方法 (因为一句话输入多个token, 所以其实对于每一次输入都会产生一套residual stream结果)
- `last`: 最后一个token输入后
- `mean`: 所有token时候那一层residual stream的平均
- `exp`: 当时想的是让AI设计一种通过一个超参数控制的加权平均 即 `a=0 平均; a=1 最后一个数字权重1.0; a=0.5 越靠近最后一个数字权重更大 且不是线性的` 实际上相比`mean`并没贡献什么明显特征

然后给这些句子 按照不同的方法 绘制出来他们之间的相似度
几乎没什么有用的 统计显著的特征.

最后的结果是几乎没有统计相关性, 有时候只看某一层看起来表示出了, 但绘制多点、多试试 就发现其实并不行

我原以为大模型可以学到某种概念的表示, 事实上其概念表示大概是像普遍相传的那样, 是黑箱的.

原本的想法是找到最显著的那一层, 然后SFT(对比学习)使得其学习相关句子不相干句子.

做到这一步, 我重申了一下那个任务, AI突然提醒我文本也可以embedding.

我傻逼了, 原来最简单的方法就是qwen3-embedding那样 直接对basemodel的输出做对比学习，可见base model是真的非常... Base....

# EXP_EMBEDDING

AI写的调用 "qwen3-embedding" 的代码 主要就是看个结果 具体实现没了解

# Condenser
真正写成比较软件的地方 这部分懒得写了