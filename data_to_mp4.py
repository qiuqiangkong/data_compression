import numpy as np
import cv2
import pickle

from utils import mu_law, inv_mu_law, analog_to_digital, digital_to_analog


def test_encode():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    tensor = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    video_path = "output.mp4"
    meta_path = "output.pkl"

    v_scales = encode_tensor_to_video(
        tensor=tensor, 
        video_path=video_path,
    )
    # v_mins: (T,)
    # v_maxs: (T,)

    ###
    meta = {
        "v_scales": v_scales
    }
    pickle.dump(meta, open(meta_path, "wb"))
    print("Write out meta to {}".format(meta_path))


def test_decode():

    video_path = "output.mp4"
    meta_path = "output.pkl"

    meta = pickle.load(open(meta_path, "rb"))
    v_scales = meta["v_scales"]

    tensor = decode_video_to_tensor(
        video_path=video_path, 
        v_scales=v_scales
    )
    # (T, W, H)

    ### (Optional) Evaluation - Compare with ground truth ###
    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    gt_tensor = load_tensor(path=path, width=width, height=height)

    error = np.mean(np.abs(tensor - gt_tensor))
    rel_error = error / np.mean(np.abs(gt_tensor))
    print("Absolute Error: {:6f}".format(error))
    print("Relative Error: {:6f}".format(rel_error))


def load_tensor(path, width, height):

    loaded = np.load(path, allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape((data.shape[0], width, height))

    return data


def encode_tensor_to_video(tensor, video_path, fps=100):

    frames_num, width, height = tensor.shape

    v_scales = np.max(np.abs(tensor), axis=(1, 2))
    
    tmp = v_scales[:, None, None]

    tensor /= tmp
    tensor = mu_law(tensor)
    tensor = analog_to_digital(tensor)

    out = cv2.VideoWriter(
        filename=video_path, 
        fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
        fps=fps, 
        frameSize=(height, width), 
        isColor=False
    )

    for n in range(frames_num):
        out.write(tensor[n])
        
    out.release()
    
    print("Write video to {}".format(video_path))

    return v_scales


def decode_video_to_tensor(video_path, v_scales):

    cap = cv2.VideoCapture(filename=video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tensor = []

    while True:

        status, x = cap.read()  # (W, H, C)

        if not status:
            break

        tensor.append(x[:, :, 0])

    tensor = np.stack(tensor, axis=0).astype(np.int32)
    # (T, W, H)

    tensor = digital_to_analog(tensor)
    tensor = inv_mu_law(tensor)
    tensor *= v_scales[:, None, None]

    return tensor


if __name__ == '__main__':

    test_encode()
    test_decode()