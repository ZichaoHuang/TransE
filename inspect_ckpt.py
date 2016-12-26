import argparse

import tensorflow as tf


def print_tensors_in_ckpt_file(file_name, tensor_name):
    """
    prints tensors in a ckpt file

    if no `tensor_name` is provided, prints the tensor names and shapes in the ckpt file
    if `tensor_name` is provided, prints the content of the tensor

    :param file_name: name of the ckpt file
    :param tensor_name: name of tensor to inspect
    """
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        if args.all_tensors is True:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print('tensor_name: ', key)
                print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode('utf-8'))
        else:
            print('tensor_name: ', tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:
        print(str(e))
        if 'corrupted compressed block contents' in str(e):
            print('it is like your ckpt file has been compressed with SNAPPY.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file_name',
        type=str,
        help='ckpt file name'
    )
    parser.add_argument(
        '--tensor_name',
        type=str,
        help='name of the tensor to inspect'
    )
    parser.add_argument(
        '--all_tensors',
        type=bool,
        default=False,
        help='if True, print the values of all the tensors'
    )
    args = parser.parse_args()

    print_tensors_in_ckpt_file(args.file_name, args.tensor_name)
