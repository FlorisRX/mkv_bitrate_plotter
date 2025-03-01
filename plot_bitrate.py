import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import numpy as np
import math
from scipy.signal.windows import gaussian

def get_video_bitrate_data(video_file, start_time=None, end_time=None):

    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_packets",
        "-select_streams", "v",
        "-of", "json",
        video_file
    ]

    try:
        print("Running FFprobe...")
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        print("Processing FFprobe results...")
        ffprobe_output = result.stdout
        ffprobe_data = json.loads(ffprobe_output)
        #print(f"FFprobe output: {ffprobe_output}")

        timestamps = []
        packet_sizes = []
        keyframes = []
        frame_index = 0

        for frame in ffprobe_data.get("packets", []):
            try:
                if frame.get("media_type") == "video" or frame.get("codec_type") == "video":
                    frame_index += 1
                    timestamp = frame.get("pts_time")

                    if timestamp is None:
                        continue
                    timestamp_float = float(timestamp)

                    packet_size = int(frame.get("size"))
                    key_frame = frame.get("key_frame") == '1'
                    
                    #print(f"Frame {frame_index}: Timestamp: {timestamp_float}, Packet Size: {packet_size}, Key Frame: {key_frame}")
                    if (start_time is None or timestamp_float >= start_time) and \
                    (end_time is None or timestamp_float <= end_time):
                        timestamps.append(timestamp_float)
                        packet_sizes.append(packet_size)
                        keyframes.append(key_frame)
                    if end_time is not None and timestamp_float > end_time:
                        print(f"End time reached. Stopping processing at frame {frame_index}.")
            except (ValueError, TypeError) as e:
                print(f"Error processing frame {frame_index}, {e}")
        if not timestamps:
            print("No video frames found in the specified time range or in the output.")
            return None

        return timestamps, packet_sizes, keyframes

    except subprocess.CalledProcessError as e:
        print(f"FFprobe command failed: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding FFprobe JSON output: {e}")
        print(f"FFprobe Output:\n{ffprobe_output}")
        return None


def moving_average(data, window_size):
    """
    Calculates the simple moving average of a data series.
    """
    if window_size > len(data):
        print("Warning: Window size is larger than data length. Returning empty smoothed data.")
        return np.array([])  # Return empty array if window is too large

    if not isinstance(data, np.ndarray):
        data = np.array(data) # Convert to numpy array for efficient operations

    smoothed_data = np.convolve(data, np.ones(window_size), 'valid') / window_size
    # 'valid' mode returns output of length len(data) - window_size + 1.
    # We will pad the beginning to match original length for plotting purposes.
    padding = np.array([np.nan] * (window_size - 1)) # Use NaN for padding so it's not plotted
    smoothed_data_padded = np.concatenate((padding, smoothed_data))
    return smoothed_data_padded

def gaussian_weighted_moving_average(data, window_size=9, std_dev=3):
    """
    Calculates the Gaussian weighted moving average of a data series.
    """
    print(f"(window size: {window_size}, std dev: {std_dev})")
    if window_size > len(data):
        print("Warning: Window size is larger than data length. Returning empty smoothed data.")
        return np.array([])  # Return empty array if window is too large
    if window_size % 2 == 0:
        print("Warning: Window size must be odd for Gaussian Weighted to be symmetric.")
        #raise ValueError("Window size must be odd for Gaussian Weighted Moving Average to be symmetric.")

    if not isinstance(data, np.ndarray):
        data = np.array(data) # Convert to numpy array for efficient operations

    window = gaussian(window_size, std_dev)
    normalized_window = window / window.sum() # Normalize to preserve sum (approximately)
    #print(f"Normalized window: {normalized_window}, std={std_dev}")
    smoothed_data = np.convolve(data, normalized_window, mode='same') # Use 'same' mode to keep output size same as input
    
    return smoothed_data

def plot_bitrate(timestamps, packet_sizes, keyframes, video_file, window_size, show_packets=False):
    if not timestamps or not packet_sizes:
        print("No data to plot.")
        return

    bitrates_mbps = [4*(size/8) / 1024 for size in packet_sizes] # Is multiplying by 4 correct? No idea why it's needed

    std_dev = window_size/6     # Higher value = smoother curve
    smoothed_bitrates_mbps = gaussian_weighted_moving_average(bitrates_mbps, window_size, std_dev)

    # Create an even smoother curve that scales with total video length (for easier visual inspection)
    window_2 = int(len(bitrates_mbps) * 0.05) # Use 5% of total frames as window size
    std_dev_2 = window_2/6
    smoothed_bitrates_mbps_2 = gaussian_weighted_moving_average(bitrates_mbps, window_2, std_dev_2)

    if not smoothed_bitrates_mbps.size: # Check if smoothing returned empty array
        print("No smoothed data to plot due to large window size.")
        return

    keyframe_timestamps = [timestamps[i] for i, flag in enumerate(keyframes) if flag]
    keyframe_bitrates_mbps = [bitrates_mbps[i] for i, flag in enumerate(keyframes) if flag]
    non_keyframe_timestamps = [timestamps[i] for i, flag in enumerate(keyframes) if not flag]
    non_keyframe_bitrates_mbps = [bitrates_mbps[i] for i, flag in enumerate(keyframes) if not flag]

    # Calculate total average bitrate from `bitrates_mbps`
    total_avg_bitrate = sum(bitrates_mbps) / len(bitrates_mbps)
    print(f"TOTAL AVG Bitrate: {total_avg_bitrate:.2f} mbps")

    plt.figure(figsize=(12, 7))

    # Plot non-keyframes as a dots
    if show_packets:
        plt.plot(timestamps, bitrates_mbps, marker='.', markersize=1, linestyle='', color='black', label='Packets')
        # plt.plot(non_keyframe_timestamps, non_keyframe_bitrates_mbps, linestyle='-', linewidth=0.5, color='blue', label='Non-Keyframes')
        # Plot keyframes as unconnected dots
        plt.plot(keyframe_timestamps, keyframe_bitrates_mbps, marker='o', markersize=5, linestyle='', color='red', label='Keyframes')
    # Plot smoothed data
    plt.plot(timestamps, smoothed_bitrates_mbps, linestyle='-', linewidth=0.1, color='green', label=f'Smoothed Bitrate ({window_size}-frame window)')
    plt.plot(timestamps, smoothed_bitrates_mbps_2, linestyle='-', linewidth=1.2, color='blue', label='Smoothing (5% window)')
    plt.xlabel("Time (seconds)")
    plt.ylabel("mbps")
    if show_packets:
        plt.ylim(0, max(bitrates_mbps) * 1.05)
    else:
        plt.ylim(0, max(smoothed_bitrates_mbps) * 1.1) # Set y-axis limit to 110% of max bitrate

    # Set x-ticks dynamically based on video length
    video_length_sec = timestamps[-1]
    raw_magnitude = math.log10(abs(video_length_sec/20))
    magnitude = int(raw_magnitude)
    round_value = 10**(magnitude) #if magnitude > 1 else 1 #if video length is less than 10s, rounding_base will be 1
    #print(f"  ****  Rounding interval: {round_value}, {raw_magnitude=}, {magnitude=} for max len = {video_length_sec}")
    x_tick_interval = video_length_sec / 20
    #print(f"  ****  X-tick interval (before rounding): {x_tick_interval}")
    x_tick_interval = round(x_tick_interval / round_value) * round_value # Round to nearest multiple of round_value
    #print(f"  ****  X-tick interval (after rounding): {x_tick_interval}")

    plt.xticks(np.arange(0, video_length_sec, step=x_tick_interval))
    plt.fill_between(timestamps, smoothed_bitrates_mbps, color='green', alpha=0.25)
    plt.title(f"Video bitrate '{video_file}'")
    plt.grid(True, alpha=0.5, linewidth=0.5, color='#777777')
    plt.legend()
    #plt.tight_layout()

    ax = plt.gca()
    # Use ScalarFormatter to turn off scientific notation on the y-axis
    formatter = ticker.ScalarFormatter(useMathText=True) # You can try False here too, depending on matplotlib version
    formatter.set_scientific(False) # Force no scientific notation
    ax.yaxis.set_major_formatter(formatter)

    plt.savefig(f"{video_file}_bitrate_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot video bitrate over time using FFprobe, separating keyframes and allowing time range selection.")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("--start", type=float, help="Start time (seconds) for plotting.")
    parser.add_argument("--end", type=float, help="End time (seconds) for plotting.")
    parser.add_argument("--window", type=int, default=9, help="Window size for smoothing (odd number).")
    parser.add_argument("--show-packets", action="store_true", help="Show individual packet sizes as dots.")
    args = parser.parse_args()

    video_file = args.video_file
    start_time = args.start
    end_time = args.end
    window_size = args.window
    show_packets = args.show_packets

    bitrate_data = get_video_bitrate_data(video_file, start_time, end_time)

    if bitrate_data:
        timestamps, packet_sizes, keyframes = bitrate_data
        #print(f"{bitrate_data}")
        plot_bitrate(timestamps, packet_sizes, keyframes, video_file, window_size, show_packets)
        print(f"Plot saved as {video_file}_bitrate_plot.png")
    else:
        print("Failed to extract bitrate data.")