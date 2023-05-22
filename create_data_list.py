from utils.data_utils import split_data_to_train_test, create_enroll_from_json
from models.speechbrain_ecpa import AttackSRModel
import numpy as np
from utils.data_utils import save_emb_as_npy
import matplotlib.pyplot as plt
from utils.data_utils import get_signal_from_wav_random
import torchaudio
# import matplotlib
from IPython.display import Audio, display
from utils.general import calculator_snr_direct
# from utils.data_utils import save_emb_as_npy
import torch
# matplotlib.use('Agg')

# should run only once when using new data to create train and test partition and corespending lists
# create enroll list using specific speaker recognition model

DATA_PATH = "data/LIBRI/d3"

def create_uniform_perturbation():
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(1, 100, 48000)
    # src_signal = get_signal_from_wav_random("../data/libri_train-clean-100/200/200-124140-0012.flac")

    uniform_noise = 0.18 * np.random.uniform(low=-1.0, high=1.0, size=(48000))
    # save_emb_as_npy()

    plt.plot(t, uniform_noise)
    plt.title('Uniform Noise')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()

def add_awgn():
    # Signal Generation
    # matplotlib inline

    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(1, 180, 48000)
    x_volts = 0.6 * np.sin(t / (2 * np.pi))
    plt.subplot(3, 1, 1)
    plt.plot(t, x_volts)
    plt.title('Signal')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.show()

    x_watts = x_volts ** 2
    plt.subplot(3, 1, 2)
    plt.plot(t, x_watts)
    plt.title('Signal Power')
    plt.ylabel('Power (W)')
    plt.xlabel('Time (s)')
    plt.show()

    x_db = 10 * np.log10(x_watts)
    plt.subplot(3, 1, 3)
    plt.plot(t, x_db)
    plt.title('Signal Power in dB')
    plt.ylabel('Power (dB)')
    plt.xlabel('Time (s)')
    plt.show()
    # Adding noise using target SNR

    # Set a target SNR
    target_snr_db = 20
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = x_volts + noise_volts

    # Plot signal with noise
    plt.subplot(2, 1, 1)
    plt.plot(t, y_volts)
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.show()
    # Plot in dB
    y_watts = y_volts ** 2
    y_db = 10 * np.log10(y_watts)
    plt.subplot(2, 1, 2)
    plt.plot(t, 10 * np.log10(y_volts ** 2))
    plt.title('Signal with noise (dB)')
    plt.ylabel('Power (dB)')
    plt.xlabel('Time (s)')
    plt.show()




def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def add_noise_awgn():
    # based on https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    t = np.linspace(1, 100, 48000)
    # max = 0.49
    file_path_men = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8063/8063-274116-0029.flac"
    # max = 0.43
    file_path_men2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/1355/1355-39947-0018.flac"
    # max = 0.365
    file_path_w = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8465/8465-246943-0012.flac"
    # max = 0.389
    file_path_w2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/32/32-21631-0008.flac"
    x_volts = get_signal_from_wav_random(file_path_men)
    # src_signal = get_signal_from_wav_random("../data/libri_train-clean-100/200/200-124140-0012.flac")
    x_volts2 = np.sin(t / (2 * np.pi))

    plot_waveform(x_volts.unsqueeze(0), 16000)

    plt.subplot(3, 1,1)

    # plt.plot(t,x_volts) # plot depend time
    plt.plot(x_volts)
    plt.title('Original Signal')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    # plt.ylim([-0.7, 0.7])
    plt.show()
    # #
    x_watts = x_volts ** 2

    # Adding noise using target SNR

    # Set a target SNR
    target_snr_db = 20

    # Calculate signal power and convert to dB
    # sig_avg_watts = torch.mean(x_watts)
    sig_avg_watts = np.mean(x_watts.numpy())
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db

    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = x_volts + noise_volts

    snr_calc = calculator_snr_direct(x_volts.unsqueeze(0),(y_volts).unsqueeze(0))
    plot_waveform(y_volts.unsqueeze(0), 16000)

    # Plot signal with noise
    plt.subplot(3, 1, 2)
    plt.plot(t, y_volts)
    plt.title('Signal with noise')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.show()

    # Plot noise
    plt.subplot(3, 1, 3)
    plt.plot(t, noise_volts)
    plt.title('Noise Only')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.show()

    # # Plot in dB
    # y_watts = y_volts ** 2
    # y_db = 10 * np.log10(y_watts)
    # plt.subplot(2, 1, 2)
    # plt.plot(t, 10 * np.log10(y_volts ** 2))
    # plt.title('Signal with noise (dB)')
    # plt.ylabel('Power (dB)')
    # plt.xlabel('Time (s)')
    # plt.show()
    #

def main():
    '''
    split data to train and test, storing train and test lists in parent path
    :return: json files
    '''
    data_root_path = DATA_PATH
    split_data_to_train_test(data_root_path)

    '''
    create enroll list using attackSR model. should define a model  
    :return: pickle and json file 
    '''
    model_instance = AttackSRModel()
    train_list_path = "data/LIBRI/data_lists/d3/train_files.json"
    create_enroll_from_json(train_list_path, model_instance)


if __name__ == '__main__':
    # create_uniform_perturbation()
    # add_awgn()
    add_noise_awgn()
    # main()
