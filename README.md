# crnn_centerloss_pytorch

pytorch crnn  with centerloss to solve the near word problem

阿里解决形近字方案： https://zhuanlan.zhihu.com/p/258135120?utm_source=wechat_timeline

1. 这里使用了两种处理流程来实现 crnn + centerloss;

	流程一：
	
	ctc loss就不介绍了，center loss 最早用于人脸识别中的loss函数，该loss 的目的就是更好的扩大类间距离，缩小类内距离；而在字符识别中的使用，可以

	分为以下几个步骤：

	step1：得到模型的预测结果，和双向长短期记忆法第二个rnn的lstm（线性层之前）的输出结果（这个其实就是用来描述一个时间片T的512维embeding特征）；

	step2：将预测结果中字符数量和标签字符数量相等的预测结果保留下来，并提取出相应的真实标签，和对应的feature的位置；

	step3：将step的feature提取出来，和对应的label，计算centerloss；

	step4：将计算出的centerloss结果乘以系数然后和ctc loss求和；

    	参考tensorflow版本 https://github.com/tommyMessi/crnn_ctc-centerloss，博客https://blog.csdn.net/qq_22764813/article/details/114134398 ; 
	
	这里只是按照tensorflow的处理流程用pytorch进行了复现，并对中间处理流程进行了添加及速度方面的优化
	
	核心处理逻辑参考crnn.py脚本，脚本里进行了相关注释，应该好理解！呼呼呼

 
	流程二： 
	
	百度 paddleOCR https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/enhanced_ctc_loss.md
	
	paddleOCR使用centerloss时，简单粗暴很多，没有像流程一那样的中间处理流程，至于为什么说简单粗暴，看如下链接paddle是怎么实现的就知道了

	https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/losses/center_loss.py 或者直接看这里的paddle_train.py脚本

	注意： paddleOCR 会使用与训练模型对centerloss的center中心进行初始化操作，具体初始化步骤第一个链接中有提及：
	
		值得一提的是， 在C-CTC Loss中，选择随机初始化Center并不能够带来明显的提升. 我们的Center初始化方法如下
		
		1. 基于原始的CTCLoss， 训练得到一个网络N

		2. 挑选出训练集中，识别完全正确的部分, 组成集合G

		3. 将G中的每个样本送入网络，进行前向计算， 提取最后一个FC层的输入（即feature）及其经过argmax计算的结果（即index）之间的对应关系

		4. 将相同index的feature进行聚合，计算平均值，得到各自字符的初始center.

	个人感觉这种方式有些麻烦，假设embedding维度为512,分类数为6000, 训练结束后会有一个 [512, 6000]的矩阵K，其实这个矩阵就是所谓的centerloss的特征

	中心，这点跟人脸识别提取人物的人脸中心特征道理一样，就不细述了, 所以，你懂得，可以直接使用这个矩阵K作为center进行初始化...


注意： 流程一与流程二中，centerloss 所占的比重不一样，流程一中λ=0.00001， 流程二中λ=0.25

2. 最终效果：最终结果在全量测试集(字典包含所有字符)上提升不太明显，大概是0.9%左右，在自己构建的形近字测试集（字典只包含形近字字符）上提升了2.3%

3. 号外号外～～～ 注意: 这里crnn.py train.py paddle_train.py 脚本中只是给出了关键调用步骤,并不能直接运行，添加到自己代码中，把相关网络进行替换就行！！！（干，别跟我说你这都嫌麻烦...）







