import os
import glob
import subprocess
from pathlib import Path
import sys
import numpy as np



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

cmd = 'ls -l'


# def runpynvml():
#     from pynvml import *
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print("Total memory:", info.total)
#     print("Free memory:", info.free)
#     print("Used memory:", info.used)

# TODO: check how to run
def gputil_decorator(func):
    def wrapper(*args, **kwargs):
        import nvidia_smi
        import prettytable as pt

        try:
            table = pt.PrettyTable(['Devices','Mem Free','GPU-util','GPU-mem'])
            nvidia_smi.nvmlInit()
            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                table.add_row([i, f"{mem.free/1024**2:5.2f}MB/{mem.total/1024**2:5.2f}MB", f"{res.gpu:3.1%}", f"{res.memory:3.1%}"])

        except nvidia_smi.NVMLError as error:
            print(error)

        print(table)
        return func(*args, **kwargs)
    return wrapper


from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def split_libri_to_train_test(src_path="data/libri_train-clean-100", dest_path ="data/train_test-librispeech_train-100"):
        # path_obj = Path(src_path)
        # filename = path_obj.stem

        # Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

        id_dir_to_train = os.listdir(src_path)

        # train_files = defaultdict()
        # validete_files = defaultdict()

        for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
            if not os.path.exists(Path(f'{dest_path}/train/{speaker_id}')):
              os.makedirs(Path(f'{dest_path}/train/{speaker_id}'))
              os.makedirs(Path(f'{dest_path}/test/{speaker_id}'))
            all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*.flac")]
            x_train, x_test = train_test_split(all_speaker_path, test_size=0.2,
                                               shuffle=True)
            print("cwd  :" ,os.getcwd() )
            for uttr in x_train:
                p = subprocess.Popen(['cp', f'{uttr}',
                                      f'{dest_path}/train/{speaker_id}/{Path(uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
                p.wait()

            for test_uttr in x_test:
                p = subprocess.Popen(['cp', f'{test_uttr}',
                                      f'{dest_path}/test/{speaker_id}/{Path(test_uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
                p.wait()


            # $$$ change to true in eval
            # train_files[speaker_id] = x_train
            # test_files[speaker_id] = x_test

        # write_json(f'{Path(src_path).parent}/{dest_path}/{filename}/train_files', train_files)
        # write_json(f'{Path(src_path).parent}/data_lists/{filename}/test_files', test_files)




    # all_dirs = os.listdir(src_path)
    # for curr in all_dirs:
    #     inner = [name for name in os.listdir(os.path.join(src_path, curr)) if
    #              os.path.isdir(os.path.join(src_path, curr, name))]
    #     for directory in inner:
    #         # os.system(f'{cmd} {directory}')
    #         p = subprocess.Popen(['rm', '-r', f'{directory}'], cwd=os.path.join(src_path, curr))
    #         p.wait()
    #     # print("in")
    #     # inner = os.listdir(os.path.join(src_path,curr))
    #     # files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.wav'), recursive=True) for lab in labels]


def split_voxceleb1_to_train_test(src_path="data/voxceleb1", dest_path ="data/train_test-voxceleb1"):
        # path_obj = Path(src_path)
        # filename = path_obj.stem

        # Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

        id_dir_to_train = os.listdir(src_path)

        # train_files = defaultdict()
        # validete_files = defaultdict()

        for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
            if not os.path.exists(Path(f'{dest_path}/train/{speaker_id}')):
              os.makedirs(Path(f'{dest_path}/train/{speaker_id}'))
              os.makedirs(Path(f'{dest_path}/test/{speaker_id}'))
            all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*/*.wav")]
            x_train, x_test = train_test_split(all_speaker_path, test_size=0.2,
                                               shuffle=True)
            print("cwd  :" ,os.getcwd() )
            for uttr in x_train:
                p = subprocess.Popen(['cp', f'{uttr}',
                                      f"{dest_path}/train/{speaker_id}/{uttr.split('/')[-2]}_{Path(uttr).name}"])#, cwd=os.path.join(src_path, speaker_id))
                p.wait()

            for test_uttr in x_test:
                p = subprocess.Popen(['cp', f'{test_uttr}',
                                      f"{dest_path}/test/{speaker_id}/{test_uttr.split('/')[-2]}_{Path(test_uttr).name}"])#, cwd=os.path.join(src_path, speaker_id))
                p.wait()


            # $$$ change to true in eval
            # train_files[speaker_id] = x_train
            # test_files[speaker_id] = x_test

        # write_json(f'{Path(src_path).parent}/{dest_path}/{filename}/train_files', train_files)
        # write_json(f'{Path(src_path).parent}/data_lists/{filename}/test_files', test_files)


def get_subset_speakers_from_dataset(src_path="data/voxceleb1", dest_path="data/sampels_test-voxceleb1", spks_num=100):

    if not os.path.exists(Path(f'{dest_path}')):
        os.makedirs(Path(f'{dest_path}'))
    speakers_dir = os.listdir(src_path)

    np.random.seed(42)
    random_speakers = np.random.choice(speakers_dir, spks_num)
    for speaker_id in tqdm(random_speakers, desc="Speaker_id:"):
        p = subprocess.Popen(['cp', '-R', f'{os.path.join(src_path,speaker_id)}', f"{dest_path}/{speaker_id}"])
        p.wait()

def main_nvidia():
# https://support.huaweicloud.com/intl/en-us/modelarts_faq/modelarts_05_0374.html
    import nvidia_smi
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(
            f"|Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")
        flush_gpu_memory = ((mem.total- mem.free)/ mem.total) * 100
        print(flush_gpu_memory)
        if flush_gpu_memory > 10:
            from numba import cuda
            device = cuda.get_current_device()
            device.reset()

    # gputil_decorator()
    # cmd = "watch -n0.1 nvidia-smi"
    # p = subprocess.Popen(['watch', '-n0.1', 'nvidia-smi'])
    # p.wait()


def remove_directories():
    src_path = "data//libri-test-clean"

    all_dirs = os.listdir(src_path)
    for curr in all_dirs:
        inner = [name for name in os.listdir(os.path.join(src_path,curr)) if os.path.isdir(os.path.join(src_path,curr,name))]
        for directory in inner:
            # os.system(f'{cmd} {directory}')
            p = subprocess.Popen(['rm', '-r',f'{directory}'], cwd=os.path.join(src_path, curr))
            p.wait()
        # print("in")
        # inner = os.listdir(os.path.join(src_path,curr))

def run_nvidia_smi_command():
    p = subprocess.Popen(['nvidia-smi'] )
    print(p)
    p.wait()

def run_kill_command():
    p = subprocess.Popen(['sudo','kill' ,'-9' ,'39739'])
    print(p)
    p.wait()


    # files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.wav'), recursive=True) for lab in labels]
import torch
if __name__ == '__main__':
    # get_subset_speakers_from_dataset()

    # split_voxceleb1_to_train_test()
    # run_nvidia_smi_command()
    # run_kill_command()
    # torch.cuda.empty_cache()
    # main_nvidia()
    # main()
    # split_libri_to_train_test()
    print("finish main")


# 'nvidia-smi'

# 39739