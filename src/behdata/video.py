import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.draw import circle
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle


def extract_frames_from_clip(video_data, istart, iend, channel=0):
    """
    Extract [istart:iend] frames from VideoFileCLip structure
    Note it only uses a single channel
    Args:
        video_data: VildeFileClip
        istart: start of chunk
        iend: end of chunk
        channel: channel color

    Returns:

    """
    # i have an i-start and i-end and video data and want to return an array
    fps = video_data.fps
    frames = np.arange(istart, iend)
    num_frames = len(frames)
    video_frames = np.zeros(([num_frames] + list(video_data.size[::-1])))

    # check state durations
    for ii, frame_idx in enumerate(frames):
        frame_idx_sec = frame_idx / fps
        video_frames[ii] = video_data.get_frame(frame_idx_sec)[:, :, channel]
    return video_frames


def get_discrete_chunks(states, include_edges=True, max_state=0):
    """
    Find occurences of each discrete state

    input:
        states: list of trials, each trial is numpy array containing discrete state for each frame
        include_edges: include states at start and end of chunk

    output:
        indexing_list: list of length discrete states, each list contains all occurences of that discrete state by [chunk number, starting index, ending index]

    """
    if max_state == 0:
        max_state = max([max(x) for x in states])
    indexing_list = [[] for x in range(max_state + 1)]

    for i_chunk, chunk in enumerate(states):

        chunk = np.pad(
            chunk, (1, 1), mode="constant", constant_values=-1
        )  # pad either side so we get start and end chunks
        split_indices = np.where(np.ediff1d(chunk) != 0)[
            0
        ]  # Don't add 1 because of start padding, this is now indice in original unpadded data
        split_indices[
            -1
        ] -= 1  # Last index will be 1 higher that it should be due to padding

        for i in range(len(split_indices) - 1):

            which_state = chunk[
                split_indices[i] + 1
                ]  # get which state this chunk was (+1 because data is still padded)

            if not include_edges:  # if not including the edges
                if split_indices[i] != 0 and split_indices[i + 1] != (
                        len(chunk) - 2 - 1
                ):
                    indexing_list[which_state].append(
                        [i_chunk, split_indices[i], split_indices[i + 1]]
                    )
            else:
                indexing_list[which_state].append(
                    [i_chunk, split_indices[i], split_indices[i + 1]]
                )

    # Convert lists to numpy arrays
    indexing_list = [
        np.asarray(indexing_list[i_state]) for i_state in range(max_state + 1)
    ]

    return indexing_list


def zoom_in_coordinates_v0(clip, x, y, rectangle_size=50):
    # deprecated for numpy compatibility
    nx, ny = clip.size
    fps = clip.fps

    def zoom_in(get_frame, t, dotsize = 2, rectangle_size=rectangle_size):
        # get frame a time t in seconds
        image = get_frame(t)

        # copy frame [ny x ny x N]
        # N is the number of 3 color channels
        frame = image.copy()

        # convert t from secs to samples
        index = int(np.round(t * 1.0 * fps))

        if index % 1000 == 0:
            print("\nTime frame @ {} [sec] is {} [idx]\n".format(t, index))

        # get centers of window: x_mean, y_mean
        xc = x[index]
        yc = y[index]

        # get window around centers
        # x: xc - r : xc + r
        xi = slice(max(xc - rectangle_size, 0), min(xc + rectangle_size, nx - 1))
        yi = slice(max(yc - rectangle_size, 0), min(yc + rectangle_size, ny - 1))

        # re-calculate center
        xc = xc - max(xc - rectangle_size, 0)
        yc = yc - max(yc - rectangle_size, 0)

        # build a frame with 3 color channels
        frame_region = np.zeros((2*rectangle_size, 2*rectangle_size, 3))

        # fill frame right corner matches right corner
        frame_region[:int(yi.stop-yi.start), :int(xi.stop-xi.start), :] = frame[yi, xi, :]

        rr, cc = circle(yc, xc, dotsize, shape=(2*rectangle_size, 2*rectangle_size))
        frame_region[rr, cc, :] = (1, 1, 0)

        return frame_region

    clip_cropped = clip.fl(zoom_in)

    return clip_cropped


def zoom_in_coordinates(clip, x, y, rectangle_size=50):
    nx, ny = clip.size
    fps = clip.fps
    # test: compare clip.duration and duration of x and y
    assert clip.duration*fps >= len(x)
    assert clip.duration*fps >= len(y)
    
    def zoom_in(get_frame, t, dotsize = 2, rectangle_size=rectangle_size):
        # get frame a time t in seconds
        image = get_frame(t)

        # copy frame [ny x ny x N]
        # N is the number of 3 color channels
        frame = image.copy()

        # convert t from secs to samples
        index = int(np.round(t * 1.0 * fps))
        #print('index {}, time {}'.format(index, t))
        if index % 1000 == 0:
            print("Time frame @ {} [sec] is {} [idx]\n".format(t, index))

        # get centers of window: x_mean, y_mean
        xc = x[index]
        yc = y[index]

        # get window around centers
        # x: xc - r : xc + r
        xi = slice(max(xc - rectangle_size, 0),
                   min(xc + rectangle_size, nx - 1))
        
        yi = slice(max(yc - rectangle_size, 0),
                   min(yc + rectangle_size, ny - 1))


        # re-calculate center
        xc = xc - max(xc - rectangle_size, 0)
        yc = yc - max(yc - rectangle_size, 0)

        # build a frame with 3 color channels
        frame_region = np.zeros((2*rectangle_size, 2*rectangle_size, 3))

        # fill frame right corner matches right corner
        yi1, yi2 = int(yi.start) ,int(yi.stop)
        xi1, xi2 = int(xi.start) ,int(xi.stop)

        frame_region[:yi2-yi1, :xi2-xi1, :] = frame[yi1:yi2, xi1:xi2, :]

        rr, cc = circle(yc, xc, dotsize, shape=(2*rectangle_size, 2*rectangle_size))
        frame_region[rr, cc, :] = (1, 1, 0)

        return frame_region

    clip_cropped = clip.fl(zoom_in)

    return clip_cropped


def make_syllables_movie(
        video_data,
        states,
        actual_K,
        n_buffer=5,
        file_name="movie_syllables.mp4",
        patch_size=(40, 40),
        plot_frame_rate=5,
        plot_n_frames=200,
        bs=20,
        n_pre_frames=3,
        scale=255,
        min_threshold=0,
        interval=20,
):
    """
    Make movie of subplots for different classes
    :param video_data:  VideoFileclip
    :param states: list of array(s)
        where each array is filled with an integer
    :param actual_K:    int
        maximum number of classes
    :param n_buffer:
    :param file_name:
    :param patch_size:
    :param plot_frame_rate:
    :param plot_n_frames:
    :param bs:
    :param n_pre_frames:
    :param scale:
    :param min_threshold:
    :param interval:
    :return:
    """
    movie_dim2, movie_dim1 = video_data.size

    state_indices = get_discrete_chunks(states, include_edges=True)

    # Get all example over thresholds
    over_threshold_instances = [[] for _ in range(actual_K)]
    for i_state in range(actual_K):
        if state_indices[i_state].shape[0] > 0:
            over_threshold_instances[i_state] = state_indices[i_state][
                (np.diff(state_indices[i_state][:, 1:3], 1) > min_threshold)[:, 0]
            ]
            np.random.shuffle(over_threshold_instances[i_state])  # Shuffle instances

    dim1 = int(np.floor(np.sqrt(actual_K)))
    dim2 = int(np.ceil(actual_K / dim1))

    # Initialize syllable movie frames
    plt.clf()
    fig_dim_div = movie_dim2 * dim2 / 10  # aiming for dim 1 being 10
    fig, axes = plt.subplots(
        dim1,
        dim2,
        figsize=((movie_dim2 * dim2) / fig_dim_div, (movie_dim1 * dim1) / fig_dim_div),
    )

    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if i < actual_K:
            ax.set_title("Syllable " + str(i+1), fontsize=15)

        else:
            ax.set_axis_off()
    fig.tight_layout(pad=0)

    # Initialize frames
    ims = [[] for _ in range(plot_n_frames + bs + 200)]

    # Loop through syllables
    for i_syllable in range(actual_K):

        print("\t Making frames for state {} ".format(i_syllable))
        if len(over_threshold_instances[i_syllable]) > 0:
            i_chunk = 0
            i_frame = 0

            while i_frame < plot_n_frames:

                # If current chunk is > number of chunks to go over
                # plot 0s screeen
                if i_chunk >= len(over_threshold_instances[i_syllable]):
                    im = fig.axes[i_syllable].imshow(
                        np.zeros((movie_dim1, movie_dim2)),
                        animated=True,
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )
                    ims[i_frame].append(im)
                    i_frame += 1
                else:

                    # ---------------
                    # Get movie chunk
                    # ---------------

                    istart = max(
                        over_threshold_instances[i_syllable][i_chunk, 1] - n_pre_frames,
                        0,
                        )
                    iend = over_threshold_instances[i_syllable][i_chunk, 2]

                    # Extract chunk
                    movie_chunk = extract_frames_from_clip(video_data, istart, iend)

                    c = 0

                    # Loop over frames in syllable chunk
                    for i in range(movie_chunk.shape[0]):
                        # add frame
                        im = fig.axes[i_syllable].imshow(
                            movie_chunk[i, :, :] / scale,
                            animated=True,
                            vmin=0,
                            vmax=1,
                            cmap="gray",
                            )

                        ims[i_frame].append(im)

                        # Add red box at start of syllable
                        if (
                                over_threshold_instances[i_syllable][i_chunk, 1]
                                >= n_pre_frames
                        ):
                            syllable_start = n_pre_frames
                        else:
                            syllable_start = over_threshold_instances[i_syllable][
                                i_chunk, 1
                            ]

                        # syllable_end = over_threshold_instances[i_syllable][i_chunk, 2]

                        # check
                        # if (i > syllable_start) and (i < (syllable_start + 2)):
                        if (i >= syllable_start):
                            rect = Rectangle(
                                (5, 5),
                                patch_size[0],
                                patch_size[1],
                                linewidth=1,
                                edgecolor="r",
                                facecolor="r",
                            )
                            im = fig.axes[i_syllable].add_patch(rect)
                            ims[i_frame].append(im)

                            c += 1
                        i_frame += 1

                    # check rectangle goes over all frames
                    assert c == np.diff(over_threshold_instances[i_syllable][i_chunk][1:3])[0]

                    # Add buffer black frames
                    for j in range(n_buffer):
                        im = fig.axes[i_syllable].imshow(
                            np.zeros((movie_dim1, movie_dim2)),
                            animated=True,
                            vmin=0,
                            vmax=1,
                            cmap="gray",
                        )
                        ims[i_frame].append(im)
                        i_frame += 1

                    i_chunk += 1

    plt.tight_layout()
    ani = ArtistAnimation(
        fig,
        [ims[i] for i in range(len(ims)) if ims[i] != []],
        interval=interval,
        blit=True,
        repeat=False,
    )
    writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist="mrw"), bitrate=-1)

    ani.save(file_name, writer=writer)

    plt.close()
    return

#%%