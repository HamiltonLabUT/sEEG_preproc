# sEEG_preproc
sEEG and ECoG preprocessing pipeline tools for the Hamilton Lab. This public release is under development for use by others. 

To calculate the high gamma analytic amplitude of your neural signal, you must first load in an MNE Raw object, then pass it to the `transformData` function in `ecog_preproc_utils.py`.

```python
raw = mne.io.read_raw_fif('...') # read your fif file
data_dir = '' # path to save the data
hgdat = transformData(raw, data_dir, band='high_gamma', notch=True, CAR=True,
					  car_chans='average', log_transform=True, do_zscore=True,
					  hg_fs=100, notch_freqs=[60,120,180], overwrite=False,
					  ch_types='eeg')
```
