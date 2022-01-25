# crnn_centerloss_pytorch
pytorch crnn  with centerloss to solve the near word problem

这里使用了两种处理流程（其实有三种，另一种懒得尝试了...）来实现 crnn + centerloss， 最终结果在全量测试集(字典包含所有字符)上提升不太明显，大概是0.9%左右，在自己构建的形近字测试集（字典只包含形近字字符）上提升了2.3%

稍后有时间会把代码整理一下给挂上

