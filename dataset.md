Dataset Description
-------------------

Overview
--------

The [dataset](https://doi.org/10.5061/dryad.dncjsxm85) used in this competition consists of 10,948 sentences spoken by a single research participant as described in [Card et al. “An Accurate and Rapidly Calibrating Speech Neuroprosthesis” (2024) _New England Journal of Medicine_](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132). For each sentence, we provide the transcript of what the participant was attempting to say, along with the corresponding time series of neural spiking activity recorded from 256 microelectrodes in speech motor cortex. The dataset contains predefined _train_, _val_, and _test_ partitions. The train and val partitions include the sentence labels, and you may repartition them if desired. The _test_ partition does not include sentence labels and will be used for competition evaluation. Your goal will be to train a model to predict spoken words from neural data using the _train_ and _val_ data splits, and then use that model to predict the words being spoken during each _test_ trial.

The dataset contains a mixture of speaking strategies and sentence corpuses (see table, below) on a block-by-block basis. A description of which blocks correspond to which corpuses and data splits can be found [here](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/data/t15_copyTaskData_description.csv). To make the competition more challenging and better approximate real-world use, we do not provide labels as to the speaking strategy for each block.

Differences from Brain-to-Text '24
----------------------------------

There are some notable differences between this dataset and the one used in the Brain-to-Text 2024 challenge:
|                     | Brain-to-Text '24                                                                                           | Brain-to-Text '25                                                                                                             |
|---------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Participant         | 'T12'                                                                                                       | 'T15'                                                                                                                         |
| Neural recordings   | 128 intracortical electrodes in speech motor cortex, 128 intracortical electrodes in inferior frontal gyrus | 256 intracortical recording electrodes in speech motor cortex                                                                 |
| Dataset period      | 25 sessions spanning 4 months                                                                               | 45 sessions spanning 20 months                                                                                                |
| Number of Sentences | 12,100                                                                                                      | 10,948                                                                                                                        |
| Sentence corpus     | Switchboard                                                                                                 | 50-word vocabulary, Switchboard, Openwebtext2, Harvard sentences, custom high-frequency word sentences, random word sentences |
| Speaking strategy   | Attempted vocalized                                                                                         | Attempted vocalized or attempted silent                                                                                       |
| Speaking rate       | ~62 words per minute                                                                                        | ~30 words per minute (attempted vocalized) or ~50 words per minute (attempted silent)                                         |


Data format
-----------

The dataset can be downloaded either from this Kaggle competition page (see below) or from [Dryad](https://doi.org/10.5061/dryad.dncjsxm85). There are 10,948 sentences from 45 sessions spanning 20 months. Data is stored in `.hdf5` files. **An example of how to load this data using the Python `h5py` library is provided on our GitHub repository, [here](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/model_training/evaluate_model_helpers.py#L29)**.

Each trial of data includes:

*   The session date, block number, and trial number
*   512 neural features (2 features \[-4.5 RMS threshold crossings and spike band power\] per electrode, 256 electrodes), binned at 20 ms resolution. The data were recorded from the speech motor cortex via four high-density microelectrode arrays (64 electrodes each). The 512 features are ordered as follows in all data files:
    *   0-64: ventral 6v threshold crossings
    *   65-128: area 4 threshold crossings
    *   129-192: 55b threshold crossings
    *   193-256: dorsal 6v threshold crossings
    *   257-320: ventral 6v spike band power
    *   321-384: area 4 spike band power
    *   385-448: 55b spike band power
    *   449-512: dorsal 6v spike band power
*   The ground truth sentence label (for train and val splits)
*   The ground truth phoneme sequence label (for train and val splits)

Loading the data in Python
--------------------------

You can load the data using the `h5py` Python library. See example code below, and also on our GitHub repository, [here](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/model_training/evaluate_model_helpers.py#L29).

    import h5py
    
    def load_h5py_file(file_path):
        data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_num': [],
            'trial_num': [],
        }
        # Open the hdf5 file for that day
        with h5py.File(file_path, 'r') as f:
    
            keys = list(f.keys())
    
            # For each trial in the selected trials in that day
            for key in keys:
                g = f[key]
    
                neural_features = g['input_features'][:]
                n_time_steps = g.attrs['n_time_steps']
                seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
                seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
                transcription = g['transcription'][:] if 'transcription' in g else None
                sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
                session = g.attrs['session']
                block_num = g.attrs['block_num']
                trial_num = g.attrs['trial_num']
    
                data['neural_features'].append(neural_features)
                data['n_time_steps'].append(n_time_steps)
                data['seq_class_ids'].append(seq_class_ids)
                data['seq_len'].append(seq_len)
                data['transcriptions'].append(transcription)
                data['sentence_label'].append(sentence_label)
                data['session'].append(session)
                data['block_num'].append(block_num)
                data['trial_num'].append(trial_num)
        return data
    

Data fields
-----------

*   `neural_features`: Temporally binned (20 ms) neural features for each trial (512 X T).
*   `n_time_steps`: Number of time steps per trial.
*   `seq_class_ids`: Integer phoneme sequence labels for each trial. Integers correspond to phonemes using the following mapping:

    LOGIT_TO_PHONEME = [
    'BLANK',    # "BLANK" = CTC blank symbol
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',    # "|" = silence token
    ]
    

*   `seq_len`: Number of phoneme labels per trial.
*   `transcriptions`: ASCII representation of sentence label for each trial.
*   `sentence_label`: Raw text sentence label for each trial.
*   `session`: Date that the trial's data was collected. Each date has a number of blocks, each block has a number of trials.
*   `block_num`: Research block number that the trial is sourced from.
*   `trial_num`: Trial number that the trial is sourced from.

More examples
-------------

You can refer to our [GitHub repository](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) for thorough examples on how to download and unzip the data, where to put it, and how to train and evaluate a baseline RNN model with it.

Files
-----

131 files

Size
----

13.45 GB

Type
----

hdf5, yaml, txt + 1 other

License
-------

[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

### data\_link.txt(39 B)

get\_app

fullscreen

chevron\_right

About this file

Link to Dryad data

https://doi.org/10.5061/dryad.dncjsxm85

Data Explorer
-------------

13.45 GB

*   arrow\_right
    
    folder
    
    t15\_copyTask\_neuralData
    
*   arrow\_right
    
    folder
    
    t15\_pretrained\_rnn\_baseline
    
*   article
    
    data\_link.txt
    

Summary
-------

arrow\_right

folder

131 files

get\_appDownload All

Download data

navigate\_nextminimize

content\_copyhelp

Download data

text\_snippet

Metadata
--------

### License

[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)