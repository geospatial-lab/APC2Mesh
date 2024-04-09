import numpy as np
import os

def save_args(args):
    file = open(os.path.join(args.log_dir, 'args.txt'), "w")
    for k, v in vars(args).items():
        if k in ['__dict__', '__weakref__', '__doc__']:
            continue
        file.write(f"{k}:\t {v}\n")
    file.close()

def read_transformation(denorm_file_abs):
    with open(denorm_file_abs) as f:
        trans = f.read().splitlines()

    scale = int(trans[0])
    centroid = np.array(trans[1])

    # print("scale: ", scale)
    # print("centroid: ", centroid)
    return scale, centroid

def filename_to_hash(file_path):
    import hashlib
    if not os.path.isfile(file_path):
        raise ValueError('Path does not point to a file: {}'.format(file_path))
    hash_input = os.path.basename(file_path).split('.')[0]
    hash = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % (2**32 - 1)
    return hash

def make_dir_for_file(file):
    file_dir = os.path.dirname(file)
    if file_dir != '':
        if not os.path.exists(file_dir):
            try:
                os.makedirs(os.path.dirname(file))
            except OSError as exc: # Guard against race condition
                raise

def call_necessary(file_in, file_out, min_file_size=0):
    """
    Check if all input files exist and at least one output file does not exist or is invalid.
    :param file_in: list of str or str
    :param file_out: list of str or str
    :param min_file_size: int
    :return:
    """

    if isinstance(file_in, str):
        file_in = [file_in]
    elif isinstance(file_in, list):
        pass
    else:
        raise ValueError('Wrong input type')

    if isinstance(file_out, str):
        file_out = [file_out]
    elif isinstance(file_out, list):
        pass
    else:
        raise ValueError('Wrong output type')

    inputs_missing = [f for f in file_in if not os.path.isfile(f)]
    if len(inputs_missing) > 0:
        print('WARNING: Input file are missing: {}'.format(inputs_missing))
        return False

    outputs_missing = [f for f in file_out if not os.path.isfile(f)]
    if len(outputs_missing) > 0:
        if len(outputs_missing) < len(file_out):
            print("WARNING: Only some output files are missing: {}".format(outputs_missing))
        return True

    min_output_file_size = min([os.path.getsize(f) for f in file_out])
    if min_output_file_size < min_file_size:
        return True

    oldest_input_file_mtime = max([os.path.getmtime(f) for f in file_in])
    youngest_output_file_mtime = min([os.path.getmtime(f) for f in file_out])

    if oldest_input_file_mtime >= youngest_output_file_mtime:
        # debug
        import time
        input_file_mtime_arg_max = np.argmax(np.array([os.path.getmtime(f) for f in file_in]))
        output_file_mtime_arg_min = np.argmin(np.array([os.path.getmtime(f) for f in file_out]))
        input_file_mtime_max = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(oldest_input_file_mtime))
        output_file_mtime_min = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(youngest_output_file_mtime))
        print('Input file {} \nis newer than output file {}: \n{} >= {}'.format(
            file_in[input_file_mtime_arg_max], file_out[output_file_mtime_arg_min],
            input_file_mtime_max, output_file_mtime_min))
        return True

    return False

