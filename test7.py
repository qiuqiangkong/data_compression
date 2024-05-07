import numpy as np
import cv2
import pickle


def add():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    tensor = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    video_high_8bit_path = "output_high_8bit.mp4"
    video_low_8bit_path = "output_low_8bit.mp4"
    meta_path = "output.pkl"

    v_mins, v_maxs = encode_tensor_to_video(
        tensor=tensor, 
        video_high_8bit_path=video_high_8bit_path,
        video_low_8bit_path=video_low_8bit_path
    )
    # v_mins: (T,)
    # v_maxs: (T,)

    ###
    meta = {
        "v_mins": v_mins,
        "v_maxs": v_maxs
    }
    pickle.dump(meta, open(meta_path, "wb"))
    print("Write out meta to {}".format(meta_path))


def add2():

    video_high_8bit_path = "output_high_8bit.mp4"
    video_low_8bit_path = "output_low_8bit.mp4"
    meta_path = "output.pkl"

    meta = pickle.load(open(meta_path, "rb"))
    v_mins = meta["v_mins"]
    v_maxs = meta["v_maxs"]

    tensor = decode_video_to_tensor(
        video_high_8bit_path=video_high_8bit_path, 
        video_low_8bit_path=video_low_8bit_path,
        v_mins=v_mins, 
        v_maxs=v_maxs
    )
    # (T, W, H)

    ### Compare with ground truth
    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    gt_tensor = load_tensor(path=path, width=width, height=height)

    error = np.mean(np.abs(tensor - gt_tensor))

    n = 5590
    np.mean(np.abs(tensor[n, 80] - gt_tensor[n, 80]))
    from IPython import embed; embed(using=False); os._exit(0)




def load_tensor(path, width, height):

    loaded = np.load(path, allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape((data.shape[0], width, height))

    return data

def encode_tensor_to_video(tensor, video_high_8bit_path, video_low_8bit_path, fps=100):

    frames_num, width, height = tensor.shape

    v_mins = np.min(tensor, axis=(1, 2))
    v_maxs = np.max(tensor, axis=(1, 2))

    mins = v_mins[:, None, None]
    maxs = v_maxs[:, None, None]

    tensor = (tensor - mins) / (maxs - mins) * 65535
    tensor = tensor.astype(np.int32)
    high_8_bit = (tensor // 256).astype("uint8")
    low_8_bit = (tensor % 256).astype("uint8")

    out_high_8bit = cv2.VideoWriter(
        filename=video_high_8bit_path, 
        fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
        fps=fps, 
        frameSize=(height, width), 
        isColor=False
    )

    out_low_8bit = cv2.VideoWriter(
        filename=video_low_8bit_path, 
        fourcc=cv2.VideoWriter_fourcc(*'MP4V'), 
        fps=fps, 
        frameSize=(height, width), 
        isColor=False
    )

    for n in range(frames_num):

        out_high_8bit.write(high_8_bit[n])
        out_low_8bit.write(low_8_bit[n])

    out_high_8bit.release()
    out_low_8bit.release()

    print("Write video to {}".format(video_high_8bit_path))
    print("Write video to {}".format(video_low_8bit_path))

    return v_mins, v_maxs


def decode_video_to_tensor(video_high_8bit_path, video_low_8bit_path, v_mins, v_maxs):

    cap_high_8bit = cv2.VideoCapture(filename=video_high_8bit_path)
    cap_low_8bit = cv2.VideoCapture(filename=video_low_8bit_path)
    
    width = int(cap_high_8bit.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_high_8bit.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tensor_high_8bit = []
    tensor_low_8bit = []

    while True:

        status, high_8_bit = cap_high_8bit.read()  # (W, H, C)
        status, low_8_bit = cap_low_8bit.read()  # (W, H, C)

        if not status:
            break

        tensor_high_8bit.append(high_8_bit[:, :, 0])
        tensor_low_8bit.append(low_8_bit[:, :, 0])

    tensor_high_8bit = np.stack(tensor_high_8bit, axis=0).astype(np.int32)
    tensor_low_8bit = np.stack(tensor_low_8bit, axis=0).astype(np.int32)
    # (T, W, H)

    # tensor = tensor_high_8bit * 256 + tensor_low_8bit
    tensor = tensor_high_8bit * 256

    from IPython import embed; embed(using=False); os._exit(0)

    v_mins = v_mins[:, None, None]
    v_maxs = v_maxs[:, None, None]
    tensor = tensor / 65535 * (v_maxs - v_mins) + v_mins

    return tensor


if __name__ == '__main__':

    add2()