import h5py
import numpy as np


def load_from_hdf5(filename):
    """Loads an HDF5 file and returns it as a dictionary with the original structure."""
    def recursively_load_dict(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If item is a group, load it as a dictionary
                result[key] = recursively_load_dict(item)
            else:  # Otherwise, load it as an array
                data = item[()]
                # Decode byte strings if necessary
                if isinstance(data, np.ndarray) and data.dtype.kind == 'S':  # Byte strings in NumPy arrays
                    data = np.array(data, dtype=str)
                elif isinstance(data, bytes):  # Single byte string
                    data = data.decode('utf-8')
                result[key] = data
        
        # Load any lists stored as attributes
        for key, item in group.attrs.items():
            result[key] = list(item) if isinstance(item, np.ndarray) else item
        
        return result
    
    with h5py.File(filename, 'r') as f:
        return recursively_load_dict(f)
    

def load_neural_data(data_path, subject_id, stimroot):
    """Load neural data and image file paths."""
    data = load_from_hdf5(data_path)
    # Meta data
    brain_area = data[subject_id]["neuron_metadata"]["brain_area"]
    ncsnr = data[subject_id]["neuron_metadata"]["ncsnr"]
    reliability = data[subject_id]["neuron_metadata"]["reliability"]
    # Display parameters
    stim_pos = data[subject_id]['trials']['stimulus_pos_deg']
    stim_size = data[subject_id]['trials']['stimulus_size_pix']
    # Response data
    resp_mat = data[subject_id]['repavg']['response_peak']  # Peak, avg response
    resp_temp_mat = data[subject_id]['repavg']['response_temporal']  # Temporal response
    stimulus_names = data[subject_id]['repavg']['stimulus_name']
    image_fps = [f"{stimroot}/{stimname.decode('utf8')}" for stimname in stimulus_names]
    return {
        'brain_area': brain_area,
        'ncsnr': ncsnr,
        'reliability': reliability,
        'stim_pos': stim_pos,
        'stim_size': stim_size,
        'resp_mat': resp_mat,
        'resp_temp_mat': resp_temp_mat,
        'image_fps': image_fps,
    }
    
    

def create_response_tensor(trials_stim_names, trials_resp_peak, rspavg_stim_names):
    """Create a 3D tensor (stimulus x neuron x trial) from trial responses.
    
    Args:
        trials_stim_names: Array of stimulus names for each trial
        trials_resp_peak: Array of peak responses for each trial (trial x neuron)
        rspavg_stim_names: Array of unique stimulus names
        
    Returns:
        resp_tensor: 3D tensor of responses (stimulus x neuron x max_trials)
        trial_counters: Number of trials per stimulus
    """
    # Create a dictionary mapping stimulus names to indices
    stim_to_idx = {name.decode('utf8'): i for i, name in enumerate(rspavg_stim_names)}
    # Initialize list to store trial counts for each stimulus
    trial_counts = np.zeros(len(rspavg_stim_names), dtype=int)
    # Count trials per stimulus
    for stim_name in trials_stim_names:
        trial_counts[stim_to_idx[stim_name.decode('utf8')]] += 1
    max_trials = trial_counts.max()
    # Initialize 3D tensor (stimulus x neuron x trial)
    resp_tensor = np.full((len(rspavg_stim_names), trials_resp_peak.shape[1], max_trials), np.nan)
    trial_counters = np.zeros(len(rspavg_stim_names), dtype=int)
    # Fill in the tensor with trial responses
    for trial_idx, (stim_name, trial_resp) in enumerate(zip(trials_stim_names, trials_resp_peak)):
        stim_idx = stim_to_idx[stim_name.decode('utf8')]
        resp_tensor[stim_idx, :, trial_counters[stim_idx]] = trial_resp
        trial_counters[stim_idx] += 1

    print(f"Response tensor shape (stimulus x neuron x trial): {resp_tensor.shape}")
    return resp_tensor, trial_counters


def load_neural_trial_resp_tensor(data_path, subject_id,):
    data = load_from_hdf5(data_path)
    trials_stim_names = data[subject_id]['trials']['stimulus_name']
    trials_resp_peak = data[subject_id]['trials']['response_peak']
    rspavg_stim_names = data[subject_id]['repavg']['stimulus_name']
    print("Trials shape:", trials_stim_names.shape, trials_resp_peak.shape)
    print("Rspavg shape:", rspavg_stim_names.shape)
    resp_tensor, trial_counters = create_response_tensor(trials_stim_names, trials_resp_peak, rspavg_stim_names)
    print("Response tensor shape:", resp_tensor.shape)
    print("Trial counters shape:", trial_counters.shape)
    print("min and max trial counters:", trial_counters.min(), trial_counters.max())
    return resp_tensor, trial_counters
