import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import kaiser, resample_poly, resample, sosfilt, hann, \
                         fftconvolve as conv
from pandas import DataFrame
from os.path import join


def tf_constant_phase(omega, phase_angle):
    H = np.exp(1j * phase_angle * np.sign(omega))
    H[omega == 0] = np.cos(phase_angle)
    return H


def ir_constant_phase(time, phase_angle, bandwidth):
    omegac = 2 * np.pi * bandwidth
    h = np.zeros_like(time)
    idx_zero = (time == 0)
    if any(idx_zero):
        h[idx_zero] = omegac / np.pi * np.cos(phase_angle)
    h[~idx_zero] = 1 / np.pi / time[~idx_zero] \
        * (np.sin(omegac * time[~idx_zero] + phase_angle)
           - np.sin(phase_angle))
    return h


def discrete_ir_constant_phase(n, phase_angle):
    idx_zero = (n == 0)
    idx_odd = ~idx_zero * (n % 2 == 1)
    h = np.zeros(len(n))
    h[idx_zero] = np.cos(phase_angle)
    h[idx_odd] = -2 / np.pi / n[idx_odd] * np.sin(phase_angle)
    return h


def periodic_constant_phase_shifter_ir(Ndft, phase_angle):
    n = np.arange(Ndft)
    h = np.zeros(Ndft)

    if Ndft % 2 == 0:
        n_odd = n[n % 2 == 1]
        h[n % 2 == 1] = 2 / Ndft / np.tan(np.pi * n_odd / Ndft)
    elif Ndft % 2 == 1:
        n_odd = n[n % 2 == 1]
        n_even_nonzero = n[(n % 2 == 0) & (n != 0)]
        h[n % 2 == 1] = 1 / Ndft / np.tan(np.pi * n_odd / 2 / Ndft)
        h[(n % 2 == 0) & (n != 0)] = \
            1 / Ndft / np.tan(np.pi * (n_even_nonzero + Ndft) / 2 / Ndft)
    h *= -np.sin(phase_angle)
    h[0] = np.cos(phase_angle)
    return h


def constant_phase_shifter(filter_order, phase_angle, beta=0, frac_delay=0):
    filter_length = filter_order + 1
    n = np.arange(-filter_order / 2, filter_order / 2 + 1) - frac_delay
    h = discrete_ir_constant_phase(n, phase_angle)
    h *= kaiser(filter_length, beta=beta)
    return n, h


def acausal_filter(x, h, idx_zero=None):
    """FIR filtering with group delay compensation.

    The convolution result is truncated to the same length as the input.
    The first sample of the output is determined by center.

    Parameters
    ----------
    x : array_like
        Input signal
    h : array_like
        FIR coefficients
    center : float, optional
        Index of the 0-th coefficient

    """
    Nx = len(x)
    Nh = len(h)
    if idx_zero is None:
        idx_zero = Nh // 2
    return conv(x, h)[idx_zero:idx_zero + Nx]


def constant_phase_shift_dft(x, phase_angle):
    """Constant phase shift performed in the DFT domain.

    Parameters
    ----------
    x : array_like
        Source signal
    phase_angle : float
        Phase angle in radians

    """
    N = len(x)
    X = np.fft.rfft(x)
    X[0] *= np.cos(phase_angle)
    X[1:-1] *= np.exp(1j * phase_angle)
    if N % 2 is 1:
        X[-1] *= np.exp(1j * phase_angle)
    elif N % 2 is 0:
        X[-1] *= np.cos(phase_angle)
    return np.fft.irfft(X, n=N)


def constant_phase_shift_nonrecursive_iir(x, n_start, n_stop, phase_angle):
    """Constant phase shift using nonrecursive IIR filter.

    The input signal is filtered with a constant phase shifter
    and a selection `[n_start, n_stop)` of the output signal is returned.
    The filter coefficients correspond to the analytic representation of
    a discrete time constant phase shifter which is acausal and infitely long.
    For the convolution, the overlap save method is used.

    Parameters
    ----------
    x : array_like
        Input signal.
    n_start : int
        First sample number of the selection
    n_stop : int
        Last (non-including) sample number of the selection
    phase_angle : float
        Phase angle in radian.

    """
    L = len(x)
    N_select = n_stop - n_start
    N = next_pow2(N_select)
    Nfft = 2 * N
    n_prepend = 2 * N - n_start % N
    n_append = 2 * N - (L + n_prepend) % N
    x_zeropad = prepend_append_zeros(x, n_prepend, n_append)
    M = int(len(x_zeropad) / N)  # number of blocks
    m0 = int((n_prepend + n_start) / N)  # index of the target block
    x_block = np.zeros(Nfft)  # input buffer
    y = np.zeros(N)  # output block

    for m in range(M):
        x_block[N:] = x_zeropad[m * N:(m + 1) * N]
        k = m0 - m
        idx = np.arange(k * N, (k + 1) * N)
        h = discrete_ir_constant_phase(idx, phase_angle)
        y += cconv(x_block, h, nfft=Nfft)[N:]
        x_block = np.roll(x_block, N)
    return y[:N_select]


def square_wave(f0, num_partials, amplitude, duration, fs, phase_angle=0,
                modal_window=None):
    """Generate square wave.

    Parameters
    ----------
    f0: float
        Fundamental frequency in Hz.
    num_partials: int
        Number of partials including the fundamental.
    amplitude: float
        Amplitude.
    duration: float
        Duration of the signal in seconds.
    fs: int
        Sampling frequency in Hz.
    phase: float, optional
        Constant phase shift in rad.
    modal_window: array_like, optional
        Modal weight of the Fourier series expansion.

    Returns
    -------
    time
    signal
    num_partials

    """
    if (2 * num_partials - 1) * f0 > 0.5 * fs:
        num_partials = np.floor(fs / f0 / 4 + 1 / 2)
    orders = 2 * np.arange(num_partials) + 1
    fourier_coefficients = 4 / np.pi / orders
    if modal_window is not None and len(modal_window) == num_partials:
        fourier_coefficients *= modal_window
    L = t2n(duration, fs=fs)
    t = np.arange(L) / fs
    s = np.sum([am * np.sin(2 * np.pi * m * f0 * t + phase_angle)
                for (am, m) in zip(fourier_coefficients, orders)], axis=0)
    return t, amplitude * s, num_partials


def fade(x, in_length, out_length=None, type='h', copy=True):
    """Apply fade in/out to a signal.

    If `x` is two-dimenstional, this works along the columns (= first
    axis).

    This is based on the *fade* effect of SoX, see:
    http://sox.sourceforge.net/sox.html

    The C implementation can be found here:
    http://sourceforge.net/p/sox/code/ci/master/tree/src/fade.c

    Parameters
    ----------
    x : array_like
        Input signal.
    in_length : int
        Length of fade-in in samples (contrary to SoX, where this is
        specified in seconds).
    out_length : int, optional
        Length of fade-out in samples.  If not specified, `fade_in` is
        used also for the fade-out.
    type : {'t', 'q', 'h', 'l', 'p'}, optional
        Select the shape of the fade curve: 'q' for quarter of a sine
        wave, 'h' for half a sine wave, 't' for linear ("triangular")
        slope, 'l' for logarithmic, and 'p' for inverted parabola.
        The default is logarithmic.
    copy : bool, optional
        If `False`, the fade is applied in-place and a reference to
        `x` is returned.

    """
    x = np.array(x, copy=copy)

    if out_length is None:
        out_length = in_length

    def make_fade(length, type):
        fade = np.arange(length) / length
        if type == 't':  # triangle
            pass
        elif type == 'q':  # quarter of sinewave
            fade = np.sin(fade * np.pi / 2)
        elif type == 'h':  # half of sinewave... eh cosine wave
            fade = (1 - np.cos(fade * np.pi)) / 2
        elif type == 'l':  # logarithmic
            fade = np.power(0.1, (1 - fade) * 5)  # 5 means 100 db attenuation
        elif type == 'p':  # inverted parabola
            fade = (1 - (1 - fade)**2)
        else:
            raise ValueError("Unknown fade type {0!r}".format(type))
        return fade

    # Using .T w/o [:] causes error: https://github.com/numpy/numpy/issues/2667
    x[:in_length].T[:] *= make_fade(in_length, type)
    x[len(x) - out_length:].T[:] *= make_fade(out_length, type)[::-1]
    return x


def db(x, *, power=False):
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def crest_factor(x, oversample=1):
    if oversample != 1:
        x = resample(x, oversample * len(x))
    return np.max(np.abs(x)) / np.mean(x**2)**0.5


def get_dBTruePeak(x):  # x.shape = (channels, samples)
    tmp = resample_poly(x / 4, up=32 * 1, down=1 * 1, axis=-1,
                        window=('kaiser', 5.0))
    tmp = tmp[:, t2n(50, fs=32 * 44100)]  # avoid evaluation of first samples
    # where filter needs to settle up
    L = 20*np.log10(4*np.max(np.abs(tmp), axis=-1))
    return L


def get_LUFS_Stereo(x, fs, mono_gated=False):  # x.shape = (2, samples)
    if fs == 48000:
        # Pre-Filter @ 48kHz
        bPRE = (1.53512485958697, -2.69169618940638, 1.19839281085285)
        aPRE = (1, -1.69065929318241, 0.73248077421585)
        # RLB-Filter @ 48kHz
        bRLB = (1, -2, 1)
        aRLB = (1, -1.99004745483398, 0.99007225036621)
    elif fs == 44100:  # redesigned by fs446: analog filter fit of the 48 kHz
        # filter, then bilinear transform to 44.1kHz
        # Pre-Filter @ 44.1kHz
        bPRE = (1.53089320672149, -2.65101332422074, 1.16905574510282)
        aPRE = (1, -1.66364666025175, 0.712582287855323)
        # RLB-Filter @ 44.1kHz
        bRLB = (1, -2, 1)
        aRLB = (1, -1.98917270016531, 0.989202007770735)
    else:
        bPRE = (1, 0, 0)
        aPRE = (1, 0, 0)
        bRLB = (1, 0, 0)
        aRLB = (1, 0, 0)
        print('! unknown fs, filter thru !')

    sos = np.array([bPRE, aPRE, bRLB, aRLB])
    sos = np.reshape(sos, (2, 6))
    y = sosfilt(sos, x, axis=-1)
    # ungated loudness, <= ITU BS.1770.3
    L = -0.691 + 10 * np.log10(np.sum(np.var(y, axis=-1)))

    if mono_gated:
        # !!! currently mono only !!!
        tG = 0.4  # gate time
        overlap = 0.75  # block overlap
        N = int(tG*fs)
        hop = int((1-overlap)*N)  # hop size
        Jg = int(len(y)/hop) + 1  # number of blocks to be considered
        lj = np.zeros((Jg, 1))  # loudness results in blocks
        zj = np.zeros((Jg, 1))

        nStart = 0
        for j in range(Jg-1):  # block hopping
            zj[j] = 1/tG * np.sum(y[nStart:nStart + N]**2 * 1/fs)
            lj[j] = -0.691 + 10*np.log10(zj[j])
            nStart = nStart + hop

        if len(y) - nStart != 0:  # if there are remaining samples
            tGlast = (len(y) - nStart) / fs
            zj[Jg-1] = 1 / tGlast * np.sum(y[nStart:]**2 * 1 / fs)
            lj[Jg-1] = -0.691 + 10*np.log10(zj[j])

        # absolute threshold
        tmp = lj > -70
        Gr = -0.691 + 10*np.log10(1/sum(tmp) * sum(zj[tmp])) - 10
        # relative threshold
        tmp = lj > Gr
        # gated loudness, >= ITU BS.1770.4
        L = -0.691 + 10*np.log10(1 / sum(tmp) * sum(zj[tmp]))

    return L


def pink_noise(nrows, ncols=16):
    """Generate pink noise using the Voss-McCartney algorithm.

    https://github.com/AllenDowney/ThinkDSP/blob/master/code/voss.ipynb

    Parameters
    ----------
    nrows : int
        Number of values to generate (signal length)
    rcols : int
        Number of random sources to add

    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


def pink_train(pulse_length, repetitions, silences, fs):

    pre_silence, silence, post_silence = silences

    L_pre = t2n(pre_silence, fs=fs, ms=True)
    L_pulse = t2n(pulse_length, fs=fs, ms=True)
    L_period = t2n(pulse_length + silence, fs=fs, ms=True)
    L_total = t2n(repetitions * pulse_length
                  + (repetitions - 1) * silence
                  + pre_silence + post_silence, fs=fs, ms=True)

    window = np.zeros(L_total)
    for i in range(repetitions):
        n_start = L_pre + i * L_period
        window[n_start:n_start + L_pulse] += hann(L_pulse)

    x = pink_noise(L_total, ncols=100)
    x -= np.mean(x)
    x *= 0.9 / np.max(np.abs(x))
    y = window * x
    return y


def normalize_truepeak(x, xmax, oversample=1):
    xpeak = np.max(np.abs(resample(x, oversample * len(x))))
    return x * xmax / xpeak


def prepend_append_zeros(x, n_prepend, n_append):
    """Prepend and append zeros to signal."""
    return np.concatenate([np.zeros(n_prepend), x, np.zeros(n_append)])


def periodic_signal(x, repetition):
    return np.tile(x, repetition)


def db2lin(x):
    """Convert decibel to linear scale."""
    return 10**(x / 20)


def t2n(t, fs, ms=False):
    """Convert time in (milli-) seconds to samples."""
    return int(np.round((0.001 if ms else 1) * t * fs))


def n2t(n, fs, ms=False):
    """Convert samples to (milli-) seconds."""
    return (1000 if ms else 1) * n / fs


def mono2multi(x, channels=2):
    """Convert mono signal to multichannels."""
    return np.tile(x, [channels, 1]).T


def next_pow2(x):
    """Next power of 2."""
    return int(2**(np.ceil(np.log2(x))))


def cconv(x1, x2, nfft=None):
    """Circular Convolution."""
    if nfft is None:
        nfft = np.max([len(x1), len(x2)])
    X1 = np.fft.rfft(x1, n=nfft)
    X2 = np.fft.rfft(x2, n=nfft)
    return np.fft.irfft(X1 * X2)


def make_periodic_stimuli(x, fs, name, phase_angles, repetition, peak_db,
                          t_period, t_predelay, t_intro, t_outro, channels=2,
                          save_stimuli=True, save_crestfactor=True,
                          out_dir=None, db_lufs_ref=-23):

    if np.isscalar(phase_angles):
        phase_angles = [phase_angles]
    if out_dir is None:
        out_dir = '.'

    t_signal = n2t(len(x), fs, ms=True)
    t_prepend = t_predelay
    t_append = t_period - t_signal - t_predelay
    n_prepend = t2n(t_prepend, fs, ms=True)
    n_append = t2n(t_append, fs, ms=True)
    n_intro = t2n(t_intro, fs, ms=True)
    n_outro = t2n(t_outro, fs, ms=True)
    peak = db2lin(peak_db)

    x = normalize_truepeak(x, peak, oversample=4)
    x = prepend_append_zeros(x, n_prepend, n_append)

    # prepare loudness normalization according to ITU.BS1770
    tmp = periodic_signal(x, repetition)
    db_lufs = get_LUFS_Stereo(tmp, fs, mono_gated=True)  # get LUFS of reference
    print('initial loudness:', db_lufs, 'LUFS')
    gain = 10**((db_lufs_ref - db_lufs)/20)  # gain towards reference LUFS

    if save_crestfactor:
        C = np.zeros_like(phase_angles)  # crest factors
    for i, phi in enumerate(phase_angles):
        y = constant_phase_shift_dft(x, phi)
        y = periodic_signal(y, repetition)

        y *= gain  # for equal loudness
        print('final loudness:', get_LUFS_Stereo(y, fs, mono_gated=True), 'LUFS')

        # Check potential clipping
        if np.amax(np.abs(y)) > 1-2**-23:  # max value for 24 Bit quantization
            raise OverflowError('error: signal amplitude larger than 1, ' +
                                'clipping in wav file will occur, ' +
                                'consider to reduce ''db_lufs_ref'' in make_periodic_stimuli()')

        y = prepend_append_zeros(y, n_intro, n_outro)
        if save_crestfactor:
            C[i] = crest_factor(y, oversample=1)
        y = mono2multi(y, channels)
        if save_stimuli:
            out_name = name + '_phi{:03.0f}'.format(np.rad2deg(phi))
            sf.write(join(out_dir, out_name + '.wav'), y, fs, subtype='PCM_24')
    if save_crestfactor:
        np.savetxt(join(out_dir, name + '_crestfactor.txt'),
                   np.stack([phase_angles, C]))
        plt.figure()
        plt.plot(phase_angles*180/np.pi, 20*np.log10(C))
        plt.xlabel(r'$\phi$  / deg')
        plt.ylabel('Crest Stimulus / dB')
        plt.title(name)
        plt.grid(True)
        plt.savefig(join(out_dir, name + '_crestfactor.png'))


def make_aperiodic_stimuli_lame(
        x, fs, name, n_start, n_stop, phase_angles, filter_order,
        t_fadein, t_fadeout, t_intro, t_outro, channels=2,
        save_stimuli=True, crestfactor=True, crestfactor_full=False,
        out_dir=None, db_lufs_ref=-23):
    """Generate Nonperiodic Stimuli using FIR constant phase shifters.

    The constant phase shifter is linearly convolved with the input signal
    for individual phase angles, which may take a long time.
    See 'make_aperiodic_stimuli()' for a equivalent but more efficient
    implementation.
    """
    if np.isscalar(phase_angles):
        phase_angles = [phase_angles]
    if out_dir is None:
        out_dir = '.'

    n_intro = t2n(t_intro, fs, ms=True)
    n_outro = t2n(t_outro, fs, ms=True)
    n_fadein = t2n(t_fadein, fs, ms=True)
    n_fadeout = t2n(t_fadeout, fs, ms=True)

    if crestfactor_full:
        C_full = np.zeros_like(phase_angles)
    if crestfactor:
        C_selection = np.zeros_like(phase_angles)

    # prepare loudness normalization according to ITU.BS1770
    db_lufs = get_LUFS_Stereo(x[n_start:n_stop], fs, mono_gated=True)  # get LUFS of reference
    print('initial loudness:', db_lufs, 'LUFS')
    gain = 10**((db_lufs_ref - db_lufs)/20)  # gain towards reference LUFS

    for i, phi in enumerate(phase_angles):

        # Phase shift
        h = constant_phase_shifter(filter_order, phi, beta=8.6)[1]
        y = acausal_filter(x, h)
        if crestfactor_full:
            C_full[i] = crest_factor(y, oversample=1)

        # Selection
        y = y[n_start:n_stop]
        y = fade(y, n_fadein, n_fadeout, 'h')
        if crestfactor:
            C_selection[i] = crest_factor(y, oversample=1)

        y *= gain  # for equal loudness
        print('final loudness:', get_LUFS_Stereo(y, fs, mono_gated=True), 'LUFS')

        # Check potential clipping
        if np.amax(np.abs(y)) > 1-2**-23:  # max value for 24 Bit quantization
            raise OverflowError('error: signal amplitude larger than 1, ' +
                                'clipping in wav file will occur, ' +
                                'consider to reduce ''db_lufs_ref'' in make_aperiodic_stimuli()')

        # Add zeros and make stereo
        y = prepend_append_zeros(y, n_intro, n_outro)
        y = mono2multi(y, channels=channels)
        out_name = name + '_phi{:03.0f}'.format(np.rad2deg(phi))

        if save_stimuli:
            sf.write(join(out_dir, out_name + '.wav'), y, fs, subtype='PCM_24')
    if crestfactor_full:
        np.savetxt(join(out_dir, name + '_crestfactor_full.txt'),
                   np.stack([phase_angles, C_full]))
        plt.figure()
        plt.plot(phase_angles*180/np.pi, 20*np.log10(C_full))
        plt.xlabel(r'$\phi$  / deg')
        plt.ylabel('Crest Full Song / dB')
        plt.title(name)
        plt.grid(True)
        plt.savefig(join(out_dir, name + '_crestfactor_full.png'))

    if crestfactor:
        np.savetxt(join(out_dir, name + '_crestfactor_selection.txt'),
                   np.stack([phase_angles, C_selection]))
        plt.figure()
        plt.plot(phase_angles*180/np.pi, 20*np.log10(C_selection))
        plt.xlabel(r'$\phi$  / deg')
        plt.ylabel('Crest Stimulus / dB')
        plt.title(name)
        plt.grid(True)
        plt.savefig(join(out_dir, name + '_crestfactor_selection.png'))


def make_aperiodic_stimuli_selection(
        x, fs, name, n_start, n_stop, phase_angles, t_fadein, t_fadeout,
        t_intro, t_outro, channels=2, save_stimuli=True, crestfactor=True,
        out_dir=None, db_lufs_ref=-23):
    """Generate Nonperiodic Stimuli using the segmented convolution.

    The returned signal is a selection [n_start, n_stop] of the constant phase
    shifted signal. The overlap-save method is used.
    """
    if np.isscalar(phase_angles):
        phase_angles = [phase_angles]
    if out_dir is None:
        out_dir = '.'

    n_intro = t2n(t_intro, fs, ms=True)
    n_outro = t2n(t_outro, fs, ms=True)
    n_fadein = t2n(t_fadein, fs, ms=True)
    n_fadeout = t2n(t_fadeout, fs, ms=True)

    if crestfactor:
        C_selection = np.zeros_like(phase_angles)

    # prepare loudness normalization according to ITU.BS1770
    db_lufs = get_LUFS_Stereo(x[n_start:n_stop], fs, mono_gated=True)  # get LUFS of reference
    print('initial loudness:', db_lufs, 'LUFS')
    gain = 10**((db_lufs_ref - db_lufs)/20)  # gain towards reference LUFS

    y0 = constant_phase_shift_nonrecursive_iir(x, n_start, n_stop, 0)
    yH = constant_phase_shift_nonrecursive_iir(x, n_start, n_stop, -0.5 * np.pi)

    for i, phi in enumerate(phase_angles):

        # Phase shift and selection
        y = np.cos(phi) * y0 - np.sin(phi) * yH
        y = fade(y, n_fadein, n_fadeout, 'h')
        if crestfactor:
            C_selection[i] = crest_factor(y, oversample=4)

        y *= gain  # for equal loudness
        print('final loudness:', get_LUFS_Stereo(y, fs, mono_gated=True), 'LUFS')

        # Check potential clipping
        if np.amax(np.abs(y)) > 1-2**-23:  # max value for 24 Bit quantization
            raise OverflowError('error: signal amplitude larger than 1, ' +
                                'clipping in wav file will occur, ' +
                                'consider to reduce ''db_lufs_ref'' in make_aperiodic_stimuli()')

        # Add zeros and make stereo
        y = prepend_append_zeros(y, n_intro, n_outro)
        y = mono2multi(y, channels=channels)
        out_name = name + '_phi{:03.0f}'.format(np.rad2deg(phi))

        if save_stimuli:
            sf.write(join(out_dir, out_name + '.wav'), y, fs, subtype='PCM_24')
    if crestfactor:
        np.savetxt(join(out_dir, name + '_crestfactor_selection.txt'),
                   np.stack([phase_angles, C_selection]))


def make_aperiodic_stimuli(
        x, fs, name, n_start, n_stop, phase_angles, filter_order,
        t_fadein, t_fadeout, t_intro, t_outro, channels=2, save_stimuli=True,
        crestfactor=True, crestfactor_full=False, out_dir=None,
        db_lufs_ref=-23):
    """Generate Nonperiodic Stimuli.

    The output is obtained by linearly combine the original signal
    and the pre-computed Hilbert transform, weighted by cos(phi) and -sin(phi),
    respectively.
    """
    if np.isscalar(phase_angles):
        phase_angles = [phase_angles]
    if out_dir is None:
        out_dir = '.'

    n_intro = t2n(t_intro, fs, ms=True)
    n_outro = t2n(t_outro, fs, ms=True)
    n_fadein = t2n(t_fadein, fs, ms=True)
    n_fadeout = t2n(t_fadeout, fs, ms=True)

    if crestfactor_full:
        C_full = np.zeros_like(phase_angles)
    if crestfactor:
        C_selection = np.zeros_like(phase_angles)

    # prepare loudness normalization according to ITU.BS1770
    db_lufs = get_LUFS_Stereo(x[n_start:n_stop], fs, mono_gated=True)  # get LUFS of reference
    print('initial loudness:', db_lufs, 'LUFS')
    gain = 10**((db_lufs_ref - db_lufs)/20)  # gain towards reference LUFS

    hH = constant_phase_shifter(filter_order, -0.5 * np.pi, beta=8.6)[1]
    y0 = x
    yH = acausal_filter(x, hH)

    for i, phi in enumerate(phase_angles):

        # Phase shift
        y = np.cos(phi) * y0 - np.sin(phi) * yH
        if crestfactor_full:
            C_full[i] = crest_factor(y, oversample=1)

        # Selection
        y = y[n_start:n_stop]
        y = fade(y, n_fadein, n_fadeout, 'h')
        if crestfactor:
            C_selection[i] = crest_factor(y, oversample=1)

        y *= gain  # for equal loudness
        print('final loudness:', get_LUFS_Stereo(y, fs, mono_gated=True), 'LUFS')

        # Check potential clipping
        if np.amax(np.abs(y)) > 1-2**-23:  # max value for 24 Bit quantization
            raise OverflowError('error: signal amplitude larger than 1, ' +
                                'clipping in wav file will occur, ' +
                                'consider to reduce ''db_lufs_ref'' in make_aperiodic_stimuli()')

        # Add zeros and make stereo
        y = prepend_append_zeros(y, n_intro, n_outro)
        y = mono2multi(y, channels=channels)
        out_name = name + '_phi{:03.0f}'.format(np.rad2deg(phi))

        if save_stimuli:
            sf.write(join(out_dir, out_name + '.wav'), y, fs, subtype='PCM_24')
    if crestfactor_full:
        np.savetxt(join(out_dir, name + '_crestfactor_full.txt'),
                   np.stack([phase_angles, C_full]))
        plt.figure()
        plt.plot(phase_angles*180/np.pi, 20*np.log10(C_full))
        plt.xlabel(r'$\phi$  / deg')
        plt.ylabel('Crest Full Song / dB')
        plt.title(name)
        plt.grid(True)
        plt.savefig(join(out_dir, name + '_crestfactor_full.png'))

    if crestfactor:
        np.savetxt(join(out_dir, name + '_crestfactor_selection.txt'),
                   np.stack([phase_angles, C_selection]))
        plt.figure()
        plt.plot(phase_angles*180/np.pi, 20*np.log10(C_selection))
        plt.xlabel(r'$\phi$  / deg')
        plt.ylabel('Crest Stimulus / dB')
        plt.title(name)
        plt.grid(True)
        plt.savefig(join(out_dir, name + '_crestfactor_selection.png'))
