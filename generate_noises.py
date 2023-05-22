import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
plt.interactive(False)
# plt.imshow(img.reshape((28, 28)))
# plt.show()
# import matplotlib
# matplotlib.use('Agg')

def plot_graph(s):
    # f = np.fft.rfftfreq(len(s))
    return plt.plot(s)[0]

def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]

def noise_psd(N, psd=lambda f: 1):
    X_white = np.fft.rfft(np.random.randn(N));
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S;
    return np.fft.irfft(X_shaped);

def noise_uniform(N, psd=lambda f: 1):
    rng = np.random.default_rng() # return [0,1)
    # get [a,b), b>a --> use (b - a) * random() + a
    norm_noise = 2 * rng.random((N,)) - 1 # [-1,1)
    X_uni = rng.random((N,))
    # X_uni = np.fft.rfft(rng.random((N,)))
    # S = psd(np.fft.rfftfreq(N))
    # # Normalize S
    # S = S / np.sqrt(np.mean(S ** 2))
    # X_shaped = X_uni * S;
    # return np.fft.irfft(X_shaped);
    return norm_noise;

# def PSDGeneratorUni(f):
#     return lambda N: noise_uniform(N, f)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)


@PSDGenerator
def white_noise(f):
    return 1;


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);


@PSDGenerator
def violet_noise(f):
    return f;


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float('inf'), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))

# @PSDGeneratorUni
# def uniform_noise(f):
#     return 1;


def plot_normal_noises():
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8), tight_layout=True)
    for G, c in zip(
            [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise],
            ['brown', 'hotpink', 'white', 'blue', 'violet']):
        plot_spectrum(G(30 * 50_000)).set(color=c, linewidth=3)
    plt.legend(['brownian', 'pink', 'white', 'blue', 'violet'])
    plt.suptitle("Colored Noise");
    plt.ylim([1e-3, None]);
    plt.show()


def plot_uniform_noise():
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8), tight_layout=True)
    for G, c in zip(
            [noise_uniform],
            ['red']):
        plot_graph(G(3 * 16000)).set(color=c, linewidth=3)
    plt.legend([ 'red'])
    plt.suptitle("Colored Noise")
    plt.ylim([-1, 1])
    plt.show()

if __name__=="__main__":
    # plot_normal_noises()
    plot_uniform_noise()
    print("finish")