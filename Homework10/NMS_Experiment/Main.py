# 导入NumPy
import numpy as np
# 添加plt图表绘制的模块
import matplotlib.pyplot as plt
# 添加绘制图形的patches模块
import matplotlib.patches as patches


def main() -> object:
    # 这是对nms算法的正确性的一个基本测试

    # 初始化测试数据
    boxes = np.array([
        [200, 200, 400, 400, 0.99],
        [220, 220, 420, 420, 0.9],
        [100, 100, 150, 150, 0.82],
        [200, 240, 400, 440, 0.5]])

    overlap = 0.6
    '''设置阈值的大小'''

    # 利用NMS算法筛选限位框
    pick = py_cpu_nms(boxes, overlap, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)  # 111”表示“1×1网格，第一子图

    # 绘制所有的矩形框
    for box in boxes:
        print(box[0], box[1])
        # 绘制RP的评分
        ax.text(box[0], box[1], box[4], fontsize=13, color='blue')
        rect = patches.Rectangle((box[0], box[3]), box[2] - box[0], box[1] - box[3], linewidth=6,
                                 edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    # 绘制所有的矩形框
    for index in pick:
        box = boxes[index]
        rect = patches.Rectangle((box[0], box[3]), box[2] - box[0], box[1] - box[3], linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    # 设置坐标轴的范围
    ax.set(xlim=[0, 600], ylim=[0, 600])
    plt.show()

    print('Processing finished!')


def py_cpu_nms(dets, thresh, max_output_size):
    """Pure Python NMS baseline.
    :return: np.array
    :param dets: 参数矩阵
    :param thresh: NMS算法的分数阈值
    :param max_output_size:
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # areas记录每个方框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score进行排序，由于argsort()函数的参数采用默认值，所以进行的是升序排序，代码“[::-1]”指将数组倒序读取（即：将数组进行翻转）
    order = scores.argsort()[::-1]
    """分数值的降序排列矩阵"""

    keep = []
    while order.size > 0:
        if len(keep) >= max_output_size:
            break
        i = order[0]
        keep.append(i)
        # [1:]的意思是从数组下标1的元素直到最后一个元素，包含第一个元素，即下标0的元素
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # ovr记录每个方框的面积
        # 其数组长度为n-1，因为可以看到下面代码是order[1:]，也就是说，不包含第一个方框
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep, np.int64)


if __name__ == '__main__':
    main()
