# RNN
## 单词向量化的理解
* 所谓单词向量化，就是将每个单词映射到一个向量当中，代码中使用的skip-garm模型是用目标词来预测上下文信息，输入一个单词，根据embedding转换对应的数字化后的向量
使其输出也是一个单词(经过ont-hot)，下面简单解释一下代码。<br>
 `generate_batch(batch_size, num_skips, skip_window)`<br>
  **batch_size**：每次训练用多少数据。<br>
  **num_skips**:标签数据。<br>
  **skip_window**:相当于输入数据(单词)上下范围。从skip_window中选取num_skips个单词来输出num_skips个数据。
```
span = 2 * skip_window + 1
buffer = collections.deque(maxlen=span)
for _ in range(span):
  data_index = 0
buffer.exteng(data[data_index:data_index + span])
data_index += span
```
<br>
  从这个代码可以看出buffer里面的数据是data_index周围span个数据（deque查了资料是一种双向队列）
  
```
for i in range(batch_size//num_skips):
  context_words = [w for w in range(span) w!=skip_window]
  words_to_use = random.sample(context_words, num_skips)
```

<br>
  `context_words`不能以自己(skip_window)为上下文，所以要排除。`words_to_use`随机从context_words中选取num_skips个单词。<br>
  `embedding=tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1,1)`初始化embedding数据，维度：[5000,128]<br>
  `embedding=tf.nn.embedding_lookup(embeddings, train_inputs)`看了文档，用途跟excel的lookup函数一样作用,这里会将train_inputs和embeddings
  进行合并，并根据对已经编好号的词按相对应的数字进行查找。<br>
## RNN以及LSTM
  * RNN中，输入序列(x1,x2,x3,...)是随着**time step**递进的，这些输入先经过U、W参数计算后得到一系列(o1,o2,o3....)输出，而且o2的计算还基于x2和o1。而且RNN的
参数共享也起到了重要作用，参数矩阵U、W、V对于每一时刻是没有变化的。其意义在于，对于文字的信息其意义可能会出现在序列上的任意位置，或者同时出现在多个位置上。
  * LSTM主要改进是有三个门控制：输入门、输出门、遗忘门，由sigmoid函数和点积操作构成的。因为经过sigmoid函数处理可以将处理的信息看成（0,1）的接收比例，0就是
全部忘记。因为早期记忆会随时间呈指数级衰减，LSTM模型在RNN原有ht基础上，增加了一个Ct来保持长期记忆，Ct是依赖遗忘门、输入门的输出进行更新。<br>
  * 实现LSTM结构可分为：**多层LSTM构建**、**输入预处理**、**LSTM循环**、**损失函数计算**、**梯度计算**<br>
1. **LSTM的构建及状态初始化**：`cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.rnn_layers,state_is_tuple=True)`这里出入参数需要是列表形式<br>
  `rnn_layers`指的是rnn层数，这个函数是将他们组合到一起。<br>
  `lstm_cell`是经过`tf.contrib.rnn.BasicLSTMCell(self.embedding, forget_bias=0.0, state_is_tuple=True)`返回。<br>
  `self.state_tensor = cell.zero_state(self.batch_size, dtype=tf.float32)`初始化state_tensor,维度为batch_size。<br>
2. **输入预处理**:`data=tf.nn.embedding_lookup(embed,self.X)`数据准备经过word2vec，将需要的dictionary,reversed_dictionary,embedding保存出来。<br>
3. **LSTM循环训练**:这里我采用`tf.nn.dynamic_rnn`进行对数据训练。<br>
  `output, self.outputs_state_tensor=tf.nn.dynamic_rnn(cell,data,initate_state=self.state_tensor)`。data数据维度[5000,128]<br>
  `seq_output_final = tf.reshape(tf.concat(output,1), [-1, self.dim_embedding])`,dim_embedding=128,output维度[num_steps][batch_size,hidden_size]<br>
4. **损失函数**: `loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y, [-1]), logits= logits)`同样是交叉熵<br>
  `var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))`标准差<br>
  `self.loss = self.loss + var_loss`总损失<br>
5. **梯度计算**:
  `grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)`这里对梯度进行了裁剪，防止梯度爆炸和消失。（这里tf.gradients返回为长度为len(tvars)的tensor列表）<br>
  `clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)`，函数返回两个参数，list_clipped，修剪后的张量，以及global_norm，一个中间计算量。这里为5<br>
  具体计算公式`list_clipped[i]=t_list[i] * clip_norm / max(global_norm, clip_norm)`,其中`global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`
  `self.optimizer = train_op.apply_gradients(zip(grads, tvars), global_step=self.global_step)`。<br>
## get_train_data()
  这里的get_train_data函数，我参照了dynamic_rnn_in_tf文件，按照代码的计算，计算的一个epoch需要的steps跟老师文档的19220steps差距很大，不知道我是哪里搞错了概念
