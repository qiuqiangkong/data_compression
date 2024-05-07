import numpy as np
import cv2
import pickle


def add():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    tensor = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    output_video_path = "output.mp4"
    output_meta_path = "output.pkl"

    v_mins, v_maxs = encode_tensor_to_video(
        tensor=tensor, output_path=output_video_path)
    # v_mins: (T,)
    # v_maxs: (T,)

    ###
    meta = {
        "v_mins": v_mins,
        "v_maxs": v_maxs
    }
    pickle.dump(meta, open(output_meta_path, "wb"))
    print("Write out meta to {}".format(output_meta_path))


def add2():

    video_path = "output.mp4"
    meta_path = "output.pkl"

    meta = pickle.load(open(meta_path, "rb"))
    v_mins = meta["v_mins"]
    v_maxs = meta["v_maxs"]

    tensor = decode_video_to_tensor(video_path=video_path, v_mins=v_mins, v_maxs=v_maxs)
    # (T, W, H)

    ### Compare with ground truth
    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    gt_tensor = load_tensor(path=path, width=width, height=height)

    error = np.mean(np.abs(tensor - gt_tensor))


    from IPython import embed; embed(using=False); os._exit(0)




def load_tensor(path, width, height):

    loaded = np.load(path, allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape((data.shape[0], width, height))

    return data

'''
def encode_tensor_to_video(tensor, output_path, fps=100):

    frames_num, width, height = tensor.shape
    
    out = cv2.VideoWriter(
        filename=output_path, 
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
        fps=fps, 
        frameSize=(height, width), 
        isColor=False
    )

    v_mins = np.min(tensor, axis=(1, 2))
    v_maxs = np.max(tensor, axis=(1, 2))

    for n in range(frames_num):

                

        x = (tensor[n] - v_mins[n]) / (v_maxs[n] - v_mins[n]) * 65535
        x = np.round(x)

        high_8_bit = (x // 256).astype("uint8")
        low_8_bit = (x % 256).astype("uint8")

        out.write(x)

    out.release()

    print("Write video to {}".format(output_path))

    return v_mins, v_maxs
'''

def encode_tensor_to_video(tensor, output_path, fps=100):

    frames_num, width, height = tensor.shape
    
    out = cv2.VideoWriter(
        filename=output_path, 
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
        fps=fps, 
        frameSize=(height, width), 
        isColor=False
    )

    v_mins = np.min(tensor, axis=(1, 2))
    v_maxs = np.max(tensor, axis=(1, 2))

    for n in range(frames_num):

                

        x = (tensor[n] - v_mins[n]) / (v_maxs[n] - v_mins[n]) * 65535
        x = np.round(x)

        high_8_bit = (x // 256).astype("uint8")
        low_8_bit = (x % 256).astype("uint8")

        out.write(x)

    out.release()

    print("Write video to {}".format(output_path))

    return v_mins, v_maxs


def decode_video_to_tensor(video_path, v_mins, v_maxs):

    cap = cv2.VideoCapture(filename=video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tensor = []

    while True:
        status, x = cap.read()  # x: (W, H, C)

        if not status:
            break

        x = x[:, :, 0]  # (W, H)
        tensor.append(x)

    tensor = np.stack(tensor, axis=0)
    # (T, W, H)

    v_mins = v_mins[:, None, None]
    v_maxs = v_maxs[:, None, None]
    tensor = tensor / 255 * (v_maxs - v_mins) + v_mins

    return tensor


if __name__ == '__main__':

    add()