import os
import threading
import time

import pandas as pd
import tensorflow as tf
import numpy as np


def split_dataframe_list_to_rows(df, target_column, separator):
    """Splits column that contains list into row per element of the list.

    Args:
      df: dataframe to split
      target_column: the column containing the values to split
      separator: the symbol used to perform the split

    Returns:
      dataframe with each entry for the target column separated,
      with each element moved into a new row.  The values in the
      other columns are duplicated across the newly divided rows.

    """

    def split_list_to_rows(row, row_accumulator, target_column, separator):
        """

        Args:
          row: 
          row_accumulator: 
          target_column: 
          separator: 

        Returns:

        """
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def get_id_character_mapping(data, columns):
    """Creating a mapping between characters and ids given dataframe.

    Args:
      data: dataframe that contains characters that need to be converted to ids
      column: a column of the dataframe that contains characters that need to be converted to ids
      columns: 

    Returns:
      id_to_character: dictionary of ids and characters
      character_to_id: dictionary of characters and ids

    """
    characters = set([])
    for column in columns:
        [characters.update(set(val)) for index, val in data[column].iteritems()]
    characters = list(sorted(characters))

    id_to_character = {i: characters[i] for i in range(len(characters))}
    character_to_id = {characters[i]: i for i in range(len(characters))}
    return id_to_character, character_to_id


def get_category_to_id_mapping(data, column):
    """Creates two mappings for id and categorical value and vice verse for given column.
    Id is a unique identifier of categorical value. Starting from 0.

    Args:
      data: dataframe that contains categorical values
      column: a column of dataframe that contains categorical values for which a mapping from categorical value
    to id is needed

    Returns:
      id_to_category: dictionary of ids and categories
      category_to_id: dictionary of categories and ids

    """
    categories = sorted(data[column].unique())
    print("There are {} unique categories".format(len(categories)))
    id_to_category = {i: categories[i] for i in range(len(categories))}
    category_to_id = {categories[i]: i for i in range(len(categories))}
    return id_to_category, category_to_id


def save_as_tfrecords_multithreaded(path, original_data, columns=["sequence"], group_by_col="Label"):
    """Provided data gets splitted in to groups and processed concurrently.
    The outcome of this is a file per group.

    Args:
      path: Location where files should be stored
      original_data: dataframe which should be converted into files
      columns: a  list of columns which should be stored as sequences (Default value = ["sequence"])
      group_by_col: a column name by which split data into groups (Default value = "Label")
    Returns:

    """
    os.makedirs(path, exist_ok=True)
    threading_start = time.time()
    coord = tf.train.Coordinator()
    threads = []
    data = original_data.groupby(group_by_col)
    for group_id in data.groups:
        if isinstance(group_id, str):
            group_name = group_id.replace(".", "_").replace("-", "_")
        elif isinstance(group_id, int):
            group_name = str(group_id)
        else:
            group_name = "_".join([str(e) for e in group_id])
        filename = os.path.join(path, group_name)
        args = (filename, data.get_group(group_id), columns)
        t = threading.Thread(target=save_as_tfrecords, args=args)
        t.start()
        threads.append(t)
    coord.join(threads)
    print("Completed all threads in {} seconds".format(time.time() - threading_start))


def to_int_feature(data):
    """
    Converts int list to tf Feature
    Args:
        data: int list to be stored in tf record

    Returns:
        tf Feature that is used in building tfrecord
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=data))


def to_float_feature(data):
    """
    Converts float list to tf Feature
    Args:
        data: float list to be stored in tf record

    Returns:
        tf Feature that is used in building tfrecord
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=data))


def save_as_tfrecords(filename, data, columns=["sequence"], extension="tfrecords"):
    """Processes a dataframe and stores data into tfrecord file

    Args:
      filename: the absolute path of the tfrecords file where data should be stored
      data: dataframe containing data will be converted into tfrecord
      columns: list of columns that should be stored as varying-length sequences (Default value = ["sequence"])
      extension: file extension
    Returns:

    """
    try:
        filename = "{}.{}".format(filename, extension)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index, row in data.iterrows():
                feature = {
                    'label': to_int_feature([row[0]])
                }
                for col_name in columns:
                    value = row[col_name]
                    if isinstance(value, int):
                        feature[col_name] = to_int_feature([value])
                    elif isinstance(value, float):
                        feature[col_name] = to_float_feature([value])
                    elif not isinstance(value, (list,)) and not isinstance (value, int) and ((value.dtype == np.float32) or (value.dtype == np.float64)):
                        feature[col_name] = to_float_feature(value)
                    else:
                        feature[col_name] = to_int_feature(value)
                        feature['length_' + col_name]:  to_int_feature([len(value)])

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        print("Data was stored in {}".format(filename))
    except Exception as e:
        print("Something went wrong went writting in to tfrecords file")
        print("Error is ", str(e))


def save_as_npy(path, original_data, columns=["Label", "sequence"], ):
    """Processes a dataframe and stores data into npy file

    Args:
      filename: the absolute path of the npy file where data should be stored
      data: dataframe containing data to be stored
      columns: list of columns that should be stored
      extension: file extension
    Returns:

    """
    os.makedirs(path, exist_ok=True)
    try:
        filename = os.path.join(path, "data.npy")
        np.save(filename, original_data[columns].values)

        print("Data was stored in {}".format(filename))
    except Exception as e:
        print("Something went wrong went writting in to npy file ({})".format(filename))
        print(e)
