# ecog_preproc_utils
#
# Liberty Hamilton last modified October 2025
#

import numpy as np
import os
import mne
from pyfftw.interfaces.numpy_fft import fft, ifft, fftfreq
from tqdm import tqdm


def auto_bands(fq_min=4.0749286538265, fq_max=200., scale=7.):
    '''Get the frequency bands of interest for the neural signal 
    decomposition. Usually these are bands between 4 and 200 Hz, log
    spaced.  These filters were originally chosen by Erik Edwards for his
    thesis work (LH). See p143 of the PDF available at
    https://faculty.washington.edu/seattle/brain-physics/theses/Edwards-thesis.pdf
    for some explanation.

    Parameters
    ----------
    fq_min : float, optional
        Minimum band *center* frequency (Hz). Default is ``4.0749286538265``.
        This value ensures compatibility with the original lab convention.
    fq_max : float, optional
        Maximum band *center* frequency (Hz). Default is ``200.0``.
    scale : float, optional
        Log2 spacing factor controlling the density of centers ("steps per
        octave"). Larger values produce more bands. Default is ``7.0``.
    
    Returns
    -------
    cts : ndarray, shape (n_bands,)
        Center frequencies for each band (Hz), monotonically increasing and
        approximately log-spaced.
    sds : ndarray, shape (n_bands,)
        Bandwidth parameters (standard deviations in Hz) for the Gaussian
        bandpass kernel used in :func:`applyHilbertTransform`.

    Notes
    -----
    - Centers are computed as ``2 ** (arange(log2(fq_min)*scale, log2(fq_max)*scale) / scale)``.
    - Bandwidths follow ``10 ** (log10(0.39) + 0.5 * log10(center))``.
      This yields broader filters at higher frequencies.

    '''
    cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
    sds = 10 ** (np.log10(.39) + .5 * (np.log10(cts)))
    return cts, sds


def applyHilbertTransform(X, rate, center, sd):
    '''Apply a Gaussian bandpass + Hilbert transform (frequency-domain).

    This function constructs an analytic signal by (1) applying a Hilbert
    transform implemented via a Heaviside kernel in the frequency domain and
    (2) applying a Gaussian bandpass filter centered at ``center`` with
    standard deviation ``sd`` (both in Hz). The computation uses FFTs from
    ``pyFFTW`` for speed.

    Parameters
    ----------
    X : array_like, shape (T,)
        1D time-series data (single channel). If you have multichannel data,
        call this function per-channel (see usage in :func:`transformData`).
    rate : float
        Sampling rate of ``X`` in Hz.
    center : float
        Center frequency of the Gaussian bandpass (Hz).
    sd : float
        Standard deviation of the Gaussian bandpass (Hz). Controls bandwidth.

    Returns
    -------
    Xc : ndarray of complex, shape (T,)
        The complex-valued analytic signal. Magnitude ``np.abs(Xc)`` is the
        band-limited envelope; ``np.angle(Xc)`` gives the instantaneous phase.
    '''
    # frequencies
    T = X.shape[-1]
    freq = fftfreq(T, 1/rate)
    # heaviside kernel
    h = np.zeros(len(freq))
    h[freq > 0] = 2.
    h[0] = 1.
    # bandpass transfer function
    k = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))
    # compute analytical signal
    Xc = ifft(fft(X)*h*k)
    return Xc


def transformData(raw, data_dir, band='high_gamma', notch=True, CAR=True,
                  car_chans='average', log_transform=True, do_zscore=True,
                  hg_fs=100, notch_freqs=[60,120,180],
                  ch_types='eeg', overwrite=False, do_plot=True):
    '''Run an end-to-end preprocessing + HilbAA extraction pipeline.

    Given an MNE ``Raw`` object containing ECoG/SEEG data, this function
    optionally applies notch filtering and common average referencing (CAR),
    then extracts band-limited analytic amplitude (HilbAA) using a set of
    Gaussian bandpass filters and a Hilbert transform implemented in the
    frequency domain. Results are optionally log-transformed (log2), z-scored
    across time per channel, resampled to ``hg_fs`` Hz, plotted, and written
    to disk under structured folder names.


    Parameters
    ----------
    raw : mne.io.BaseRaw
        The input continuous data. Bad channels in ``raw.info['bads']`` are
        respected only in the sense that they are printed/logged; this
        function does not drop them automatically.
    data_dir : str or path-like
        Root directory where outputs are written. Subdirectories will be
        created as needed (e.g., ``HilbAA_70to150_8band``).
    band : {"high_gamma", "alpha", "theta", "delta", "beta", "gamma", "broadband"}, optional
        Band family to process. ``"high_gamma"`` (70–150 Hz) triggers HilbAA
        averaging across that range; other named bands use their canonical
        ranges defined internally; ``"broadband"`` processes 4–200 Hz with
        many sub-bands.
    notch : bool, optional
        If ``True``, apply notch filtering at ``notch_freqs`` first and save an
        intermediate FIF file.
    CAR : bool, optional
        If ``True``, apply common average referencing via
        ``raw.set_eeg_reference(car_chans)`` and save an intermediate FIF.
    car_chans : str or list, optional
        Channel(s) used for CAR. The default ``"average"`` uses the average of
        all EEG/ECoG channels. You may also pass a specific list of channel
        names.
    log_transform : bool, optional
        If ``True``, take ``log2`` of the magnitude of the analytic signal 
        before z-scoring. This can make values more normally distributed.
    do_zscore : bool, optional
        If ``True``, z-score each channel across time after band averaging.
    hg_fs : int, optional
        Output sampling rate (Hz) for the HilbAA signal. Default is ``100``.
    notch_freqs : list of int, optional
        Frequencies (Hz) for notch filtering (e.g., mains and harmonics).
        Default is ``[60, 120, 180]``.
    ch_types : str, optional
        Channel type passed to ``mne.create_info`` when building the output
        RawArray. Typically ``"eeg"`` for ECoG contacts.
    overwrite : bool, optional
        Whether to overwrite existing FIF files when saving.
    do_plot : bool, optional
        If ``True``, show diagnostic PSD/trace plots interactively at key
        steps. Set to ``False`` in batch or headless environments.


    Returns
    -------
    transformed_data : mne.io.Raw
        The final HilbAA ``Raw`` object, resampled to ``hg_fs`` and containing
        one channel per input contact.
    '''

    valid_bands = {'delta','theta','alpha','beta','gamma','high_gamma','broadband'}
    if band not in valid_bands:
        raise ValueError(f"band must be one of {sorted(valid_bands)}; got {band!r}")

    # The suffix that will be added to the file name as
    # different procedures occur
    full_suffix = ''

    bads = raw.info['bads']
    print('Original bad channels:')
    print(bads)
    raw.load_data()
    raw.pick_types(meg=False, eeg=True, ecog=True) 

    band_ranges = {'delta': [0.1, 4],
                   'theta': [4, 8],
                   'alpha': [8, 15],
                   'beta':  [15, 30],
                   'gamma': [30, 70],
                   'high_gamma': [70, 150]}

    if notch:
        full_suffix += '_notch'
        newfile = os.path.join(data_dir, 'Raw', f'ecog_raw{full_suffix}.fif')
        if os.path.isfile(newfile):
            print(f'{newfile} already exists, loading...')
            raw = mne.io.read_raw_fif(newfile)
        else:
            print("Doing notch filter")
            if do_plot:
                raw.plot_psd()
            raw.notch_filter(notch_freqs)
            if do_plot:
                raw.plot_psd()
                raw.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                     title='notch filtered raw data')
            try:
                #raw.info['bads'] = bads
                newfile = os.path.join(data_dir, 'Raw', f'ecog_raw{full_suffix}.fif')
                raw.save(newfile, overwrite=overwrite)
            except Exception as e:
                print(f"Couldn't save {newfile}: {e}. Do you need overwrite=True?")

    if CAR:
        full_suffix += '_car'
        newfile = os.path.join(data_dir, 'Raw', f'ecog_raw{full_suffix}.fif')
        if os.path.isfile(newfile):
            print(f'{newfile} already exists, loading...')
            raw = mne.io.read_raw_fif(newfile)
        else:
            print("Doing CAR on")
            print(car_chans)
            raw.set_eeg_reference(car_chans)
            if do_plot:
                raw.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                     title='after referencing (CAR)')
            try:
                #raw.info['bads'] = bads
                raw.save(newfile, overwrite=overwrite)
            except Exception as e:
                print(f"Couldn't save {newfile}: {e}. Do you need overwrite=True?")

    # Get center frequencies and standard deviations of the bands
    # for the Hilbert transform
    raw.load_data()
    print("Getting frequency bands for Hilbert transform")
    if band != 'delta':
        cts, sds = auto_bands()
    else:
        cts, sds = auto_bands(fq_min=0.1, fq_max=4)

    if band == 'high_gamma':
        print(f"Getting {band} band data")
        hg_dir = os.path.join(data_dir, 'HilbAA_70to150_8band')
        if not os.path.isdir(hg_dir):
            print("Creating directory %s" %(hg_dir))
            os.mkdir(hg_dir)

        # determine size of our high gamma band
        f_low = band_ranges[band][0]
        f_high = band_ranges[band][1]

        sds = sds[(cts>=f_low) & (cts<=f_high)]
        cts = cts[(cts>=f_low) & (cts<=f_high)]

    elif (band == "alpha") or (band=="theta") or (band=="delta") or (band=="beta") or (band=="gamma"):
        f_low = band_ranges[band][0]
        f_high = band_ranges[band][1]

        out_dir = os.path.join(data_dir, f'{band}_{f_low}to{f_high}')
        if not os.path.isdir(out_dir):
            print("Creating directory %s" %(out_dir))
            os.mkdir(out_dir)

        fname = f'ecog_{band}_{f_low}to{f_high}{full_suffix}.fif'

        raw_filt = raw.copy()
        print(f"Filtering data in {band} band from {f_low} to {f_high} Hz")
        print("")
        print("***********************ATTENTION!******************************************")
        print("Note that this will *not* use the analytic amplitude like high gamma")
        print("and that will be done separately")
        print("***************************************************************************")
        print("")
        raw_filt.filter(l_freq=f_low, h_freq=f_high)
        try:
            raw_filt.save(os.path.join(out_dir, fname), overwrite=overwrite)
        except Exception as e:
            print(f"Couldn't save {fname}: {e}, do you need to set overwrite=True?")
        if do_plot:
            raw_filt.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                 title=f'after filtering in {band} band')
        
        transformed_data = raw.copy()

        sds = sds[(cts>=f_low) & (cts<=f_high)]
        cts = cts[(cts>=f_low) & (cts<=f_high)]

        # Get the HilbAA data also (power within that band)
        nband = len(cts)
        hg_dir = os.path.join(data_dir, f'HilbAA_{band}_{f_low}to{f_high}_{nband}band')
        if not os.path.isdir(hg_dir):
            print("Creating directory %s" %(hg_dir))
            os.mkdir(hg_dir)   

    elif (band == "broadband"):
        f_low = 4
        f_high = 200
        hg_dir = os.path.join(data_dir, f'HilbAA_{f_low}to{f_high}_40band')
        if not os.path.isdir(hg_dir):
            print("Creating directory %s" %(hg_dir))
            os.mkdir(hg_dir)

    # do the Hilbert transform on all the data

    hg_info = mne.create_info(raw.info['ch_names'], raw.info['sfreq'], ch_types='eeg')
    if log_transform:
        full_suffix+='_log'

    print("Getting the raw data array")
    raw_data = raw.get_data()
    nchans = raw.info['nchan']
    dat = []
    
    print(f'Using data from {f_low} to {f_high}')
    print(sds, cts)
    nband = len(cts)
    for i, (ct, sd) in enumerate(tqdm(zip(cts, sds), 'applying Hilbert transform...', total=len(cts))):
        hilbdat = np.zeros((raw_data.shape))
        for ch in np.arange(nchans):
            hilbdat[ch,:] = applyHilbertTransform(raw_data[ch,:], raw.info['sfreq'], ct, sd)
        if log_transform:
            print(f"Taking log transform of {band}")
            band_data = np.log2(np.abs(hilbdat.real.astype('float32') + 1j*hilbdat.imag.astype('float32')))
        else:
            print("Not taking log transform")
            band_data = np.abs(hilbdat.real.astype('float32') + 1j*hilbdat.imag.astype('float32'))
        dat.append(band_data)
        if band == "broadband":
            fname = f'{hg_dir}/ecog_hilbAA_{f_low}to{f_high}_{nband}band_{i:02d}{full_suffix}.fif'
            hg_signal_band = (band_data - np.expand_dims(np.nanmean(band_data, axis=1), axis=1) )/np.expand_dims(np.nanstd(band_data, axis=1), axis=1)
            hgdat = mne.io.RawArray(hg_signal_band, hg_info)
            if raw.annotations: # if we rejected something reject it in HG also
                for annotation in raw.annotations:
                    # Add annotations from raw to hg data
                    onset = (annotation['onset']-(raw.first_samp/raw.info['sfreq'])) # convert start time for clinical data   
                    duration = annotation['duration']
                    description = annotation['description']
                    hgdat.annotations.append(onset,duration,description)
            print(f'Resampling from {raw.info["sfreq"]} to {hg_fs}')
            hgdat.resample(hg_fs)
            hgdat.save(fname, overwrite=overwrite)

    # hilbmat is now the analytic amplitude matrix
    hilbmat = np.array(np.hstack((dat))).reshape(dat[0].shape[0], -1, dat[0].shape[1])
    
    # average across relevant bands
    print("Taking the mean across %d bands"%(hilbmat.shape[1]))
    hg_signal = hilbmat.mean(1) # Get the average across the relevant bands 
    
    if do_zscore:
        # Z-score
        print("Z-scoring signal")
        hg_signal = (hg_signal - np.expand_dims(np.nanmean(hg_signal, axis=1), axis=1) )/np.expand_dims(np.nanstd(hg_signal, axis=1), axis=1)
    #hg_info = mne.create_info(raw.info['ch_names'], raw.info['sfreq'], ch_types)

    hgdat = mne.io.RawArray(hg_signal, hg_info)
    if raw.annotations: # if we rejected something reject it in HG also
        for annotation in raw.annotations:
            # Add annotations from raw to hg data
            onset = (annotation['onset']-(raw.first_samp/raw.info['sfreq'])) # convert start time for clinical data   
            duration = annotation['duration']
            description = annotation['description']
            hgdat.annotations.append(onset,duration,description)

    #hgdat.info['bads'] = bads
    hgdat.resample(hg_fs)
    
    fname = f'ecog_hilbAA_{f_low}to{f_high}_{nband}band{full_suffix}.fif'
    new_fname = os.path.join(hg_dir, fname) 
    print(f"Saving to {new_fname}")
    try:
        hgdat.save(new_fname, overwrite=overwrite)
    except Exception as e:
        print(f"Couldn't save {newfile}: {e}. Do you need overwrite=True?")
    transformed_data = hgdat.copy()
    if do_plot:
        hgdat.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
               title=f'after analytic amplitude in {band} band')

    return transformed_data

