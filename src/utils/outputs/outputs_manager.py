import os
import numpy as np
import pandas as pd


def get_output_links(link_path):
    """
    Get a list of link ids to be output.
    """

    try:
        # load the output links as a numpy array
        output_links = np.loadtxt(link_path, dtype = int)

    except:
        print('Unable to load output links file.')
        quit()

    return output_links



def get_output_states(state_path):
    """
    Get a list of output states to be output.
    """

    try:
        # load the output states as a list
        with open(state_path) as f:
            lines = f.readlines()
        output_states = [line.strip() for line in lines]

    except:
        print('Unable to load output states file.')
        quit()

    return output_states


def get_output_timestamps(timestamp_path):

    # check if the timestamp is all
    if timestamp_path == "all":

        output_timestamp = "all"

        return output_timestamp

    # check if this parameter is a integer
    elif isinstance(timestamp_path, int):
        # set up the time step for output
        output_timestamps = int(timestamp_path)

        return output_timestamps

    else:
        # try to load it as a file of timestamps
        output_timestamps = pd.read_csv(timestamp_path, header=None)
        # convert it to unix time
        global_start_time = pd.Timestamp('1970-01-01 00:00')
        # compute the duration difference between
        duration_difference = pd.to_datetime(output_timestamps[0]) - global_start_time
        # get a vector of unix timestamps
        unix_timestamp_list = []
        for time_duration in duration_difference:
            curr_unix_timestamp = int(time_duration.total_seconds())
            unix_timestamp_list.append(curr_unix_timestamp)
        unix_timestamp_list = np.array(unix_timestamp_list)

        return unix_timestamp_list


