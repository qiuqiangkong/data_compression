import numpy as np
import matplotlib.pyplot as plt

def get_density(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["density"]

    return data

def get_sound_speed(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["sound_speed"]

    return data

def get_pressure(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]

    return data

def get_dx(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["dx"]

    return data


def get_dy(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["dy"]

    return data


def get_Nx(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["Nx"]

    return data

def get_Ny(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["Ny"]

    return data


def get_dt(path):

    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["dt"]

    return data

def get_Nt(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["Nt"]

    return data


if __name__ == "__main__":

    path = "complicated_room1_save1.npy"


    pressure = get_pressure(path)

    sound_speed = get_sound_speed(path)

    density = get_density(path)

    

    print(pressure)
    print(type(pressure))

    print(sound_speed)
    print(type(sound_speed))

    print(density)
    print(type(density))

    # plt.subplot(211)
    # plt.imshow(sound_speed)
    # plt.subplot(212)
    # plt.imshow(density)
    # plt.show()

    plt.matshow(pressure[100].reshape(128, 128), origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)
    
