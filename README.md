# 说明
***版本要求
-----------
tensorflow >= 1.1
***


该代码使用 Seq2Seq 模型对四则运算进行了序列学习。
其中运算符包含加（+）、减（-）、乘（*）。

该Seq2Seq模型使用了双向LSTM模型作为encode，并且使用Dropout层来防止过拟合。

使用方式
---------------
输入数据：

    train：1111-22
    label：1089


使用方式：

    python generate_data.py 生成相应数据

    python seq2seq_model.py 训练模型

    python predict_model.py 预测数据



