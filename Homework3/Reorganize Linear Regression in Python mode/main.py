import numpy as np


def inference(w, b, x):        # inference, test, predict, same thing. Run model after training
    """推理函数。

        用权值w和b进行推理计算y。

        Args:
            w: 模型的权重。
            b:
            x: 样本值。

        Returns:
            返回推理计算的预测值。
    """
    pred_y = w * x + b
    return pred_y
# inference()函数结束


def eval_loss(w, b, x_list, y_list):
    """损失评价函数.

        用权值w和b进行推理计算y.

        Args:
            w: 模型的权重.
            b:
            x_list: 样本值.
            y_list:预测值.

        Returns:
            返回推理计算的预测值.
    """
    loss = 0.5*np.sum((inference(w, b, x_list)-y_list) ** 2)/x_list.size  # loss function

    return loss


def gradient(pred_y, y, x):
    diff = pred_y - y
    dw = np.sum(diff * x)/x.size
    db = diff.sum()/x.size
    return dw, db
# gradient()函数结束。


def cal_step_gradient(batch_x_list, batch_y_list, w, b, lr):
    pred_y = inference(w, b, batch_x_list)  # get label data
    dw, db = gradient(pred_y, batch_y_list, batch_x_list)

    w -= lr * dw
    b -= lr * db
    return w, b
# cal_step_gradient()函数结束。


def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0

    for i in range(max_iter):
        batch_idxs = np.random.choice(x_list.size, batch_size)
        batch_x = x_list[batch_idxs]
        batch_y = gt_y_list[batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))
    # 训练结束
# train()函数结束。


def gen_sample_data():
    """样本生成函数.

            用随机函数生成样本..

            Args:

            Returns:
                返回生成的样本.
    """
    w = np.random.randint(0, 10) + np.random.random()		# for noise random.random[0, 1)
    b = np.random.randint(0, 5) + np.random.random()
    num_samples = 100

    # 使用NumPy直接生成随机数组
    x_list = np.random.rand(num_samples)*100
    y_list = w * x_list + b + np.random.random() * 2 - 1

    return x_list, y_list, w, b


if __name__ == '__main__':
    x_list, y_list, w, b = gen_sample_data()
    # 此时获得的w和b则是模型的真实值
    # gen_sample_data()函数测试完成，没有语法错误。

    # 模拟一次训练的过程
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.0005613
    # lr = 0.00001
    max_iter = 20000
    # max_iter = 200
    # x_list = np.arange(100)
    # y_list = np.arange(100)+2
    train(x_list, y_list, 50, lr, max_iter)

    print('Processing finished!')
    print('w0:{0}, b0:{1}'.format(w, b))
    # 一元线性回归实验完成！
