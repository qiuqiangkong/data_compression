import numpy as np
import pickle
import torch
from einops import rearrange
import numpy as np

import torch_dct as dct
import matplotlib.pyplot as plt



def load_tensor(path, width, height):

    loaded = np.load(path, allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape((data.shape[0], width, height))

    return data
    


def add2():

    x = np.arange(36).reshape((6, 6))

    x = torch.Tensor(x)

    # y = rearrange(x, '(w nw) (h nh) -> w nw h nh', w=3, h=3)
    y = rearrange(x, '(nw w) (nh h) -> nw nh w h', w=2, h=2)
    
    norm = "ortho"
    a1 = dct.dct_2d(y, norm=norm)
    b1 = dct.idct_2d(a1, norm=norm)


    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    import numpy as np
    import cv2
    size = (128, 128)
    duration = 2
    fps = 25
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for _ in range(fps * duration):
        data = np.random.randint(0, 256, size, dtype='uint8')
        # data = np.zeros(size, dtype='uint8')
        data = np.arange(128*128, dtype=uint8)%256
        out.write(data)
    out.release()


# def add3():

#     import numpy as np
#     import cv2
#     size = (128, 128)
#     duration = 2
#     fps = 500
#     # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
#     out = cv2.VideoWriter('output.mkv', cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'FFV1'), fps, (size[1], size[0]), params=[
#         cv2.VIDEOWRITER_PROP_DEPTH,
#         cv2.CV_16U,
#         cv2.VIDEOWRITER_PROP_IS_COLOR,
#         0,  # false
#     ],)
#     for i in range(fps * duration):
#         data = (i + np.arange(128*128).reshape(128,128)) % 10000
#         data = data.astype('uint16')
#         out.write(data)
#     out.release()


def add3():

    import numpy as np
    import cv2
    size = (128, 128)
    duration = 2
    fps = 500
    out = cv2.VideoWriter('output.mkv', cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'FFV1'), fps, (size[1], size[0]), params=[
        cv2.VIDEOWRITER_PROP_DEPTH,
        cv2.CV_16U,
        cv2.VIDEOWRITER_PROP_IS_COLOR,
        0,  # false
    ],)
    for i in range(fps * duration):
        data = (i + np.arange(128*128).reshape(128,128)) % 10000
        data = data.astype('uint16')
        out.write(data)
    out.release() 


def add4():

    import cv2
    import numpy as np

    # cap = cv2.VideoCapture('output.mp4')
    cap = cv2.VideoCapture('output.mkv', apiPreference=cv2.CAP_FFMPEG,
    params=[
        cv2.CAP_PROP_CONVERT_RGB,
        0,  # false
    ],)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint16'))
    buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint16'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    
    cap.release()

    from IPython import embed; embed(using=False); os._exit(0)


###
def add5():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128
    data = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    import numpy as np
    import cv2
    size = (128, 128)
    # duration = 2
    fps = 100
    out = cv2.VideoWriter('output.mkv', cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'FFV1'), fps, (size[1], size[0]), params=[
        cv2.VIDEOWRITER_PROP_DEPTH,
        cv2.CV_16U,
        cv2.VIDEOWRITER_PROP_IS_COLOR,
        0,  # false
    ],)

    v_maxs = np.max(data, axis=(1, 2))
    v_mins = np.min(data, axis=(1, 2))

    for n in range(data.shape[0]):
        x = (data[n] + 10) / 20 * 65535
        # x = (data[n] - v_mins[n]) / (v_maxs[n] - v_mins[n]) * 255
        x = x.astype('uint16')
        out.write(x)
    out.release()

    from IPython import embed; embed(using=False); os._exit(0)


def add6():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128
    data = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    v_maxs = np.max(data, axis=(1, 2))
    v_mins = np.min(data, axis=(1, 2))

    import cv2

    # cap = cv2.VideoCapture('output.mp4')
    cap = cv2.VideoCapture('output.mkv', apiPreference=cv2.CAP_FFMPEG,
    params=[
        cv2.CAP_PROP_CONVERT_RGB,
        0,  # false
    ],)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint16'))
    # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    tensor = []

    while True:
        
        status, x = cap.read()
        
        if not status:
            break

        tensor.append(x)
    
    cap.release()

    tensor = np.stack(tensor, axis=0).astype(np.int32)
    
    for n in range(tensor.shape[0]):
        # buf[n, :, :] = buf[n, :, :] / 255 * (v_maxs[n] - v_mins[n]) + v_mins[n]
        tensor[n, :, :] = tensor[n, :, :] / 65535 * 20 - 10

    for n in range(100):
        print(n, np.mean(np.abs(tensor[n] - data[n])))

    error = np.mean(np.abs(tensor - data))
    from IPython import embed; embed(using=False); os._exit(0)


def add7():

    y = int_to_bool_list(15)
    from IPython import embed; embed(using=False); os._exit(0)

def mu_law(x, mu=100):
    
    output = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    return output



if __name__ == '__main__':

    add8()