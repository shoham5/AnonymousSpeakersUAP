import os
import glob
import subprocess
from pathlib import Path
import sys
import numpy as np
import random

import pandas as pd

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

def copy_spk(speaker_id, src_path, dest_path):

    # dest = os.path.join(src_path,"train", str(speaker_id))
    if not os.path.exists(Path(f'{dest_path}/train')):
        os.makedirs(Path(f'{dest_path}/train/'))
        os.makedirs(Path(f'{dest_path}/test/'))

    # all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*.flac")]

    # print("cwd  :", os.getcwd())

    p = subprocess.Popen(['cp', '-R', f'{os.path.join(src_path,"train", str(speaker_id))}', f"{dest_path}/train/"])
    p.wait()


    p = subprocess.Popen(['cp', '-R', f'{os.path.join(src_path,"test", str(speaker_id))}', f"{dest_path}/test/"])
    p.wait()


        # p.wait()


def split_libri_to_train_test_subsets(src_path="data/libri_close_set_110spk/110/", dest_path ="data/libri_close_set_110spk", num_of_spk=110):
# data/libri_close_set_110spk/110
        id_dir_to_train = os.listdir(f'{src_path}/train')
        # id_dir_to_train = list(map(int, id_dir_to_train))
        id_dir_to_train.sort()


        id_dir_to_test = os.listdir(f'{src_path}/test')
        # id_dir_to_test = list(map(int, id_dir_to_test))
        id_dir_to_test.sort()

        txt_path = "speakers_st"
        src_gender = "data/train_test-librispeech_train-100"
        males_file = f'{src_gender}/{txt_path.lower()}_males.csv'
        df_males = pd.read_csv(males_file)
        male_speakers_set = list(df_males['IN'].values)
        male_speakers_set = list(map(str, male_speakers_set))
        male_speakers_set.sort()
        subsets = [10, 20, 50, 80, 100]

        for subset in subsets:
            curr_data_set = subset # subsets[0]
            max_spk_in = subset / 2 #subsets[0] / 2
            cnt_male = 0
            cnt_female = 0

            for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
                if speaker_id in male_speakers_set:
                    cnt_male += 1
                    if cnt_male <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/{curr_data_set}')

                else:
                    cnt_female += 1
                    if cnt_female <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/{curr_data_set}')

                if cnt_female == max_spk_in and cnt_male == max_spk_in:
                    break




        # for spk in id_dir_to_train:
        #
        #
        #
        #
        # speakers_dirs = set(os.listdir(current_dir))
        # speakers_dirs = list(map(int, speakers_dirs))
        # all_speakers = len(speakers_dirs)
        #
        # count_males = [1 if i in speakers_set else 0 for i in speakers_dirs]
        #
        #
        #
        # id_dir_to_train = id_dir_to_train[:num_of_spk]
        # # train_files = defaultdict()
        # # validete_files = defaultdict()
        #
        #
        # for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
        #     if not os.path.exists(Path(f'{dest_path}/train/{speaker_id}')):
        #       os.makedirs(Path(f'{dest_path}/train/{speaker_id}'))
        #       os.makedirs(Path(f'{dest_path}/test/{speaker_id}'))
        #     all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*.flac")]
        #     x_train, x_test = train_test_split(all_speaker_path, test_size=0.2,
        #                                        shuffle=True)
        #     print("cwd  :" ,os.getcwd() )
        #     for uttr in x_train:
        #         p = subprocess.Popen(['cp', f'{uttr}',
        #                               f'{dest_path}/train/{speaker_id}/{Path(uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
        #         p.wait()
        #
        #     for test_uttr in x_test:
        #         p = subprocess.Popen(['cp', f'{test_uttr}',
        #                               f'{dest_path}/test/{speaker_id}/{Path(test_uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
        #         p.wait()
        #

            # $$$ change to true in eval
            # train_files[speaker_id] = x_train
            # test_files[speaker_id] = x_test

        # write_json(f'{Path(src_path).parent}/{dest_path}/{filename}/train_files', train_files)
        # write_json(f'{Path(src_path).parent}/data_lists/{filename}/test_files', test_files)



def split_voxceleb_to_train_test_subsets(src_path="data/train_test-voxceleb1", dest_path ="data/train_test-voxceleb1", num_of_spk=110):
# data/libri_close_set_110spk/110
# "data/train_test-voxceleb1"
        id_dir_to_train = os.listdir(f'{src_path}/train')
        # id_dir_to_train = list(map(int, id_dir_to_train))
        id_dir_to_train.sort()


        id_dir_to_test = os.listdir(f'{src_path}/test')
        # id_dir_to_test = list(map(int, id_dir_to_test))
        id_dir_to_test.sort()

        txt_path = "speakers_st"
        src_gender = "data/train_test-voxceleb1"
        males_file = f'{src_gender}/{txt_path.lower()}_males.csv'
        df_males = pd.read_csv(males_file)
        male_speakers_set = list(df_males['IN'].values)
        male_speakers_set = list(map(str, male_speakers_set))
        male_speakers_set.sort()
        subsets = [50]

        for subset in subsets:
            curr_data_set = subset # subsets[0]
            max_spk_in = subset / 2 #subsets[0] / 2
            cnt_male = 0
            cnt_female = 0

            for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
                if speaker_id in male_speakers_set:
                    cnt_male += 1
                    if cnt_male <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/{curr_data_set}')

                else:
                    cnt_female += 1
                    if cnt_female <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/{curr_data_set}')

                if cnt_female == max_spk_in and cnt_male == max_spk_in:
                    break




        # for spk in id_dir_to_train:
        #
        #
        #
        #
        # speakers_dirs = set(os.listdir(current_dir))
        # speakers_dirs = list(map(int, speakers_dirs))
        # all_speakers = len(speakers_dirs)
        #
        # count_males = [1 if i in speakers_set else 0 for i in speakers_dirs]
        #
        #
        #
        # id_dir_to_train = id_dir_to_train[:num_of_spk]
        # # train_files = defaultdict()
        # # validete_files = defaultdict()
        #
        #
        # for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
        #     if not os.path.exists(Path(f'{dest_path}/train/{speaker_id}')):
        #       os.makedirs(Path(f'{dest_path}/train/{speaker_id}'))
        #       os.makedirs(Path(f'{dest_path}/test/{speaker_id}'))
        #     all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*.flac")]
        #     x_train, x_test = train_test_split(all_speaker_path, test_size=0.2,
        #                                        shuffle=True)
        #     print("cwd  :" ,os.getcwd() )
        #     for uttr in x_train:
        #         p = subprocess.Popen(['cp', f'{uttr}',
        #                               f'{dest_path}/train/{speaker_id}/{Path(uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
        #         p.wait()
        #
        #     for test_uttr in x_test:
        #         p = subprocess.Popen(['cp', f'{test_uttr}',
        #                               f'{dest_path}/test/{speaker_id}/{Path(test_uttr).name}'])#, cwd=os.path.join(src_path, speaker_id))
        #         p.wait()
        #

            # $$$ change to true in eval
            # train_files[speaker_id] = x_train
            # test_files[speaker_id] = x_test

        # write_json(f'{Path(src_path).parent}/{dest_path}/{filename}/train_files', train_files)
        # write_json(f'{Path(src_path).parent}/data_lists/{filename}/test_files', test_files)



def split_libri_to_train_test_by_gender_subsets(src_path="data/libri_close_set_110spk/110/", dest_path ="data/libri_close_set_110spk/gender", num_of_spk=110):
# data/libri_close_set_110spk/110
        id_dir_to_train = os.listdir(f'{src_path}/train')
        # id_dir_to_train = list(map(int, id_dir_to_train))
        id_dir_to_train.sort()


        id_dir_to_test = os.listdir(f'{src_path}/test')
        # id_dir_to_test = list(map(int, id_dir_to_test))
        id_dir_to_test.sort()

        txt_path = "speakers_st"
        src_gender = "data/train_test-librispeech_train-100"
        males_file = f'{src_gender}/{txt_path.lower()}_males.csv'
        df_males = pd.read_csv(males_file)
        male_speakers_set = list(df_males['IN'].values)
        male_speakers_set = list(map(str, male_speakers_set))
        male_speakers_set.sort()
        subsets = [50]#, 20, 50, 80, 100]

        for subset in subsets:
            curr_data_set = subset # subsets[0]
            max_spk_in = subset #  / 2 #subsets[0] / 2
            cnt_male = 0
            cnt_female = 0

            for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
                if speaker_id in male_speakers_set:
                    cnt_male += 1
                    if cnt_male <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/males/{curr_data_set}')

                else:
                    cnt_female += 1
                    if cnt_female <= max_spk_in:
                        copy_spk(speaker_id, src_path, f'{dest_path}/females/{curr_data_set}')

                if cnt_female == max_spk_in and cnt_male == max_spk_in:
                    break


def split_test_by_gender_libri(src_path="data/libri-test-clean/", dest_path ="data/libri_test_open-set_gender", num_of_spk=110):


        id_dir_to_test = os.listdir(f'{src_path}')
        id_dir_to_test.sort()

        txt_path = "speakers_st_test"
        src_gender = "data/train_test-librispeech_train-100"
        males_file = f'{src_gender}/{txt_path.lower()}_males.csv'
        # males_file = f'{src_gender}/speakers_st_train_full.csv'
        df_males = pd.read_csv(males_file)
        male_speakers_set = list(df_males['IN'].values)
        male_speakers_set = list(map(str, male_speakers_set))
        male_speakers_set.sort()

        for speaker_id in tqdm(id_dir_to_test, desc="Speaker_id:"):
            if speaker_id in male_speakers_set:
                if not os.path.exists(Path(f'{dest_path}/males/{speaker_id}')):
                    os.makedirs(Path(f'{dest_path}/males/{speaker_id}'))
                p = subprocess.Popen(
                    ['cp', '-R', f'{os.path.join(src_path, str(speaker_id))}', f"{dest_path}/males/"])
                p.wait()
            else:
                if not os.path.exists(Path(f'{dest_path}/females/{speaker_id}')):
                    os.makedirs(Path(f'{dest_path}/females/{speaker_id}'))
                p = subprocess.Popen(
                    ['cp', '-R', f'{os.path.join(src_path, str(speaker_id))}', f"{dest_path}/females/"])
                p.wait()







def split_libri_to_train_test(src_path="data/libri_train-clean-100", dest_path ="data/libri_close_set_110spk/100", num_of_spk=110):


        # train_test-librispeech_train-100
        # path_obj = Path(src_path)
        # filename = path_obj.stem

        # Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

        id_dir_to_train = os.listdir(src_path)
        id_dir_to_train.sort()
        id_dir_to_train = id_dir_to_train[:num_of_spk]
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


def split_voxceleb1_to_train_test(src_path="data/voxceleb1",
                                  dest_path ="data/voxceleb1_close_set_110spk",
                                  num_of_spk=110):
        # path_obj = Path(src_path)
        # filename = path_obj.stem

        # Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

        id_dir_to_train = os.listdir(src_path)
        id_dir_to_train.sort()
        id_dir_to_train = id_dir_to_train[:num_of_spk]
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

def check_gender_dist(current_dir = "data/libri_close_set_100spk/train"):

    src_path = "data/train_test-librispeech_train-100"
    txt_path = "speakers_st"

    # voxceleb
    current_dir = "data/train_test-voxceleb1/train"
    src_path = "data/train_test-voxceleb1/"

    males_file = f'{src_path}/{txt_path.lower()}_males.csv'
    df_males = pd.read_csv(males_file)
    speakers_dirs=set(os.listdir(current_dir))
    # speakers_dirs = list(map(int, speakers_dirs))
    all_speakers = len(speakers_dirs)
    speakers_set = set(list(df_males['IN'].values))
    count_males = [1 if i in speakers_set else 0 for i in speakers_dirs ]
    males = np.array(count_males).sum()
    # females = speakers_dirs - set(list(df_males['IN'].values))
    print(f'females:  {all_speakers - males}  |  males { males}')

    # df_test_males = pd.read_csv(f'{src_path}/speakers_info_test_males.csv')
    # df_test_females = pd.read_csv(f'{src_path}/speakers_info_test_females.csv')
    # count_test_males = [1 if i in set(list(df_test_males['IN'].values)) else 0 for i in speakers_dirs ]
    # count_test_females = [1 if i in set(list(df_test_females['IN'].values)) else 0 for i in speakers_dirs]
    # print("count_test_males: ", count_test_males)
    # print("count_test_females: ", count_test_females)


def text_to_csv_libri(src_path="data/train_test-librispeech_train-100", txt_path="speakers_st"):
    cols_n = ['IN','S','SEX','S1','SUBSET','S2','MIN','S3','NAME','IDX']
    df1 = pd.read_csv(f'{src_path}/speakers_st_full1.csv',skiprows = 1,header = None,names=cols_n)
    df2 = pd.read_csv(f'{src_path}/speakers_less_full.csv',skiprows = 1,header = None,names=cols_n)
    df_full = pd.concat([df1, df2])
    df_full.reset_index()

    # df_full['IDX'] = range(1, len(df_full) + 1)
    # df_full.set_index('IDX')


    # df = pd.read_fwf(f'{src_path}/{txt_path}.txt',header=None, converters={0:str, 1:str,2:str,3:str,4:str,5:str,6:str,7:str} )
    # df['IDX'] = range(1, len(df) + 1)
    # df.set_index('IDX')
    dr_train100 = df_full[df_full['SUBSET'] == 'test-clean']
    # dr_train100 = df_full[df_full['SUBSET'] == 'train-clean-100']
    dr_train100.reset_index()
    # dr_train100.drop('IDX',axis=1,inplace=True)
    # dr_train100['IDX'] = range(1, len(dr_train100) + 1)
    # dr_train100.reset_index()
    # dr_train100.set_index('IDX')
    dr_train100 = dr_train100[['IN', 'SEX', 'SUBSET', 'MIN', 'NAME','IDX']]
    # dr_train100.to_csv(f'{src_path}/{txt_path.lower()}_train100.csv')
    dr_train100.to_csv(f'{src_path}/{txt_path.lower()}_test.csv')
    # df_full.reset_index()

    df_males = dr_train100[dr_train100['SEX'] == 'M']
    df_males.reset_index()
    df_females = dr_train100[dr_train100['SEX'] == 'F']
    df_females.reset_index()
    df_males.to_csv(f'{src_path}/{txt_path.lower()}_test_males.csv')
    df_females.to_csv(f'{src_path}/{txt_path.lower()}_test_females.csv')
    all_speakers = list(dr_train100['IN'].values)
    all_speakers.sort()
    subset_spk = all_speakers[:100]

    # df_new = pd.read_csv(f'{src_path}/{txt_path}.csv')
    print("wait")
#     /sise/home/hanina/speaker_attack/data/train_test-librispeech_train-100/


def text_to_csv_voxceleb(src_path="data/train_test-voxceleb1", csv_path="SPEAKERS_INFO"):
    cols_n = ['IN','NAME','SEX','Nationality','SUBSET']
    df_full = pd.read_csv(f'{src_path}/{csv_path}.csv',sep='\t',skiprows = 1,header = None,names=cols_n)
    print("wait")
    # df2 = pd.read_csv(f'{src_path}/speakers_less_full.csv',skiprows = 1,header = None,names=cols_n)
    # df_full = pd.concat([df1, df2])
    # df_full.reset_index()

    df_full['IDX'] = range(1, len(df_full) + 1)
    df_full.set_index('IDX')
    dr_train = df_full[df_full['SUBSET'] == 'test']
    dr_train.reset_index()
    df_males = dr_train[dr_train['SEX'] == 'm']
    df_males.reset_index()
    df_females = dr_train[dr_train['SEX'] == 'f']
    df_females.reset_index()
    df_males.to_csv(f'{src_path}/{csv_path.lower()}_test_males.csv')
    df_females.to_csv(f'{src_path}/{csv_path.lower()}_test_females.csv')
    all_speakers = list(dr_train['IN'].values)
    all_speakers.sort()
    print("wait")

    # df = pd.read_fwf(f'{src_path}/{txt_path}.txt',header=None, converters={0:str, 1:str,2:str,3:str,4:str,5:str,6:str,7:str} )
    # df['IDX'] = range(1, len(df) + 1)
    # df.set_index('IDX')


    # dr_train100 = df_full[df_full['SUBSET'] == 'train-clean-100']
    # dr_train100.reset_index()

    # dr_train100.drop('IDX',axis=1,inplace=True)
    # dr_train100['IDX'] = range(1, len(dr_train100) + 1)
    # dr_train100.reset_index()
    # dr_train100.set_index('IDX')

    # dr_train100 = dr_train100[['IN', 'SEX', 'SUBSET', 'MIN', 'NAME','IDX']]
    # dr_train100.to_csv(f'{src_path}/{txt_path.lower()}_train100.csv')
    # df_full.reset_index()

    # df_males = dr_train100[dr_train100['SEX'] == 'M']
    # df_males.reset_index()
    # df_females = dr_train100[dr_train100['SEX'] == 'F']
    # df_females.reset_index()
    # df_males.to_csv(f'{src_path}/{txt_path.lower()}_males.csv')
    # df_females.to_csv(f'{src_path}/{txt_path.lower()}_females.csv')
    # all_speakers = list(dr_train100['IN'].values)
    # all_speakers.sort()
    # subset_spk = all_speakers[:100]

    # df_new = pd.read_csv(f'{src_path}/{txt_path}.csv')
    print("wait")
#     /sise/home/hanina/speaker_attack/data/train_test-librispeech_train-100/

def create_npy_from_csv(npy_src='VOX_labels1.npy', labels_csv='labels_vox_ecapa.csv', src_path ="data/sampels_test-voxceleb1"):
    df = pd.read_csv(f'{src_path}/labels_vox_ecapa.csv')
    df = df.drop(['idx'], axis=1)
    df_dict  = df.to_dict()
    reverse_df_dict = {}
    np_labels_updated = {}
    del df
    # rev_df_dict = {v:k for k,v in df_dict['spk_id'].values()}
    for k in df_dict['spk_id'].keys():
        reverse_df_dict[df_dict['spk_id'][k]] = k

    np_labels = np.load(f'{src_path}/VOX_labels1.npy',allow_pickle=True).item()
    for spk_key in np_labels.keys():
        np_labels_updated[spk_key] = reverse_df_dict.get(spk_key.split('/')[2], 'None')
        # np_labels.update([spk_key,reverse_df_dict.get(spk_key.split('/')[2], 'None' )])
    print("wait")
    np.save(os.path.join(src_path, "VOX_labels_ecapa.npy"), np_labels_updated)


def create_scp_file_to_train_test(src_path="data/train_test-voxceleb1/50", dest_path ="data/sampels_test-voxceleb1"):
        # path_obj = Path(src_path)
        # filename = path_obj.stem

        # Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

        id_dir_to_train = os.listdir(f'{src_path}/train')
        id_dir_to_test = os.listdir(f'{src_path}/test')

        # train_files = defaultdict()
        # validete_files = defaultdict()
        all_files_train = []
        all_files_test = []
        labels = {}
        speakers = {}
        spk_cnt = 0
        for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
            all_speaker_paths = [os.path.relpath(f, Path(src_path).parent) for f in glob.glob(f"{src_path}/train/{speaker_id}/*.wav")]
            all_files_train.extend(all_speaker_paths)
            speakers[speaker_id] = spk_cnt
            spk_cnt += 1
            for uttr in all_speaker_paths:
                labels[uttr] = speakers[speaker_id]

        random.shuffle(all_files_train)
        # print(all_files)
        with open(os.path.join(dest_path,'train.scp'),'w') as f:
            for f_path in all_files_train:
                f.write(f'{f_path}\n')
                # f.write("\n")

        for speaker_id in tqdm(id_dir_to_test, desc="Speaker_id:"):
            all_speaker_paths = [os.path.relpath(f, Path(src_path).parent) for f in
                                 glob.glob(f"{src_path}/test/{speaker_id}/*.wav")]
            all_files_test.extend(all_speaker_paths)
            for uttr in all_speaker_paths:
                labels[uttr] = speakers[speaker_id]

        # random.shuffle(all_files_test)
        # print(all_files)
        with open(os.path.join(dest_path, 'test.scp'), 'w') as f:
            for f_path in all_files_test:
                f.write(f'{f_path}\n')

            np.save(os.path.join(dest_path, "VOX_labels.npy"), labels)

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

def flat_dataset_dirs(src_path="data/voxceleb1_test_open_set", dest_path="data/voxceleb1_test_open-set", spks_num=100):

    if not os.path.exists(Path(f'{dest_path}')):
        os.makedirs(Path(f'{dest_path}'))
    speakers_dir = os.listdir(src_path)

    for speaker_id in tqdm(speakers_dir, desc="Speaker_id:"):
        if not os.path.exists(Path(f'{dest_path}/{speaker_id}')):
            os.makedirs(Path(f'{dest_path}/{speaker_id}'))
        all_speaker_path = [f for f in glob.glob(f"{src_path}/{speaker_id}/*/*.wav")]
        print("cwd  :", os.getcwd())
        for uttr in all_speaker_path:
            p = subprocess.Popen(['cp', f'{uttr}',
                                  f"{dest_path}/{speaker_id}/{uttr.split('/')[-2]}_{Path(uttr).name}"])  # , cwd=os.path.join(src_path, speaker_id))
            p.wait()

    # np.random.seed(42)
    # random_speakers = np.random.choice(speakers_dir, spks_num)
    # for speaker_id in tqdm(random_speakers, desc="Speaker_id:"):
    #     p = subprocess.Popen(['cp', '-R', f'{os.path.join(src_path,speaker_id)}', f"{dest_path}/{speaker_id}"])
    #     p.wait()

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

def extend_uap_gan(root_path='/sise/home/hanina/speaker_attack/data/uap_perturbation/',file_name='Gan_uap.npy',perturb_size=48000):
    file_path = os.path.join(root_path, Path(file_name).stem)
    np_uap = np.load(f'{file_path}.npy')
    # src_signal_size = signal.shape
    # print("signal.shape[1]: ",signal.shape[1])

    c_size = math.ceil(perturb_size / np_uap.shape[0])
    # TODO: adding +1 to handle ValueError: high <= 0
    cat_size = [np_uap for i in range(c_size)]
    perturb_uap = torch.cat(cat_size, 1)
    # if signal.shape[1]==48000:
    #     print(f"signal len: {signal.shape} _ wav_path: {wav_path} _original_size:{src_signal_size} ")
    np.save(f'{root_path}/extend_{file_name}', perturb_uap.cpu())
import torch
if __name__ == '__main__':
    # text_to_csv()
    # split_libri_to_train_test()
    # split_libri_to_train_test_subsets()
    # split_libri_to_train_test_by_gender_subsets()
    # text_to_csv_voxceleb()
    # split_voxceleb_to_train_test_subsets()
    # split_libri_to_train_test_by_gender_subsets()

    # text_to_csv_libri()
    # split_test_by_gender_libri()

    create_npy_from_csv()

    # extend_uap_gan()
    # check_gender_dist()
    # get_subset_speakers_from_dataset()
    # create_scp_file_to_train_test()
    # flat_dataset_dirs()
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