import numpy as np
import numpy.matlib
import cv2
import scipy.signal as signal

# 这是“中值滤波”的作业。


# 仅支持单通道图像数据
def medianBlur(img, kernel, padding_way):
    m, n = img.shape  # 获取输入图片的尺寸（行和列）
    m_kernel, n_kernel = kernel.shape
    # 赵老师说，中值滤波模板的长度和宽度，可以不同，而且可以是偶数，感觉很神奇，第一次听说。
    img_Padding = np.zeros((m+m_kernel-1, n+n_kernel-1))
    # mStart表示在填充图中原始图像数据的起始行号。
    mStart = m_kernel >> 1
    nStart = n_kernel >> 1
    img_Padding[mStart:mStart+m, nStart:nStart+n] = img

    # cv2.imshow('img_Padding1', img_Padding)
    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()
    # # 这是Bug的测试代码

    if padding_way != 'ZERO':
        img_Padding = np.lib.pad(img, ((mStart,), (nStart,)), mode='edge')
    # 填充操作结束

    print(img_Padding)

    img_Blur = np.zeros((m, n))

    # 滑动窗口进行滤波
    for i in range(m):
        for j in range(n):
            window = kernel*img_Padding[i:i+m_kernel, j: j+n_kernel]
            img_Blur[i, j] = getMedian(window)
    # 滤波结束

    img_Blur = np.array(img_Blur).astype('uint8')
    return img_Blur
# medianBlur函数结束


def getMedian(window):
    window = np.sort(window.ravel())
    size = window.size

    # 如果window中有奇数个数字
    median = window[size >> 1]

    if size % 2 == 0:
        median = (median+window[size >> 1]) >> 1
    # 判断结束

    return median
# getMedianKernel函数结束

def getMedianKernel(m, n):
    return np.ones((m, n))
# getMedianKernel函数结束


# 首先是基本的正确性测试
img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = getMedianKernel(3, 3)
img_Blur = medianBlur(img, kernel, 'REPLICA')
print(img_Blur)

# 对图片进行测试
img_Gray = cv2.imread('./noise3.png', 0)
kernel = getMedianKernel(3, 3)
img_Blur = medianBlur(img_Gray, kernel, 'REPLICA')

print('Processing finished!')

cv2.imshow('Median Blur', img_Gray)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

cv2.imshow('Median Blur2', img_Blur)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

