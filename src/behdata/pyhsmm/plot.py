import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


color_names = [
    "red",
    "windows blue",
    "medium green",
    "dusty purple",
    "orange",
    "amber",
    "clay",
    "pink",
    "greyish",
    "light cyan",
    "steel blue",
    "forest green",
    "pastel purple",
    "mint",
    "salmon",
    "dark brown",
]
colors = sns.xkcd_palette(color_names)


def plot_vector_field_dynamics(
    Aks,
    bks,
    comb_obs=None,
    xlim=(-3, 3),
    ylim=(-3, 3),
    sharey=True,
    sharex=True,
    nxpts=20,
    nypts=20,
    FIGURE_STORE=False,
    OUTDIR="",
    fname=None,
):
    if comb_obs is None:
        comb_obs = [(0, 1)]

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    fig, ax = plt.subplots(
        len(comb_obs),
        len(Aks),
        figsize=(len(Aks) * 4, len(comb_obs) * 4),
        sharey=sharey,
        sharex=sharex,
    )

    if np.ndim(ax) == 1:
        ax = ax[None, :]

    # State indices the columns
    for state, (Ak, bk) in enumerate(zip(Aks, bks)):
        # Select pair of coordinates
        for pair_id, cpair in enumerate(comb_obs):
            As = Ak[np.ix_(cpair, cpair)]
            bs = bk[np.ix_(cpair)]
            dydt_m = xy.dot(As.T) + bs.T - xy

            ax[pair_id, state].quiver(
                xy[:, 0],
                xy[:, 1],
                dydt_m[:, 0],
                dydt_m[:, 1],
                color=colors[state % len(colors)],
                headwidth=5.0,
            )
            # ax[pair_id, state].set_title(
            #    "$A_{} x_{{t, {}{}}} + b_{} - x_t$".format(state + 1,
            #                                               cpair[0],
            #                                               cpair[1],
            #                                               state + 1)
            # )

            ax[pair_id, state].set_title(
                "$A_{} x_t + b_{} - x_t$".format(state + 1, state + 1)
            )

            ax[pair_id, state].set_xlabel("$x_{{t, {} }}$".format(cpair[0]))
            # share y
            ax[pair_id, 0].set_ylabel("$x_{{t, {} }}$".format(cpair[1]))
    plt.tight_layout()

    if FIGURE_STORE:
        if fname is None:
            fname = "Vector Field Dynamics.pdf"
        plt.savefig(os.path.join(OUTDIR, fname))
    else:
        plt.show()
    plt.close()
    return


def plot_vector_field_dynamics_data(
        Aks,
        bks,
        data,
        states,
        comb_obs=None,
        sharey=True,
        sharex=True,
        FIGURE_STORE=False,
        OUTDIR="",
        fname=None,
):

    # single data set
    if comb_obs is None:
        comb_obs = [(0, 1)]

    fig, ax = plt.subplots(
        len(comb_obs),
        len(Aks),
        figsize=(len(Aks) * 4, len(comb_obs) * 4),
        sharey=sharey,
        sharex=sharex,
    )

    if np.ndim(ax) == 1:
        ax = ax[None, :]

    # State indices the columns
    for state, (Ak, bk) in enumerate(zip(Aks, bks)):
        # Select pair of coordinates
        x_t = data[states == state, :]

        if not np.any(x_t):
            print('No data with states {}'.format(state))
            continue

        for pair_id, cpair in enumerate(comb_obs):
            As = Ak[np.ix_(cpair, cpair)]
            bs = bk[np.ix_(cpair)]

            xy = x_t[:, cpair]
            dydt_m = xy.dot(As.T) + bs.T - xy

            ax[pair_id, state].quiver(
                xy[:, 0],
                xy[:, 1],
                dydt_m[:, 0],
                dydt_m[:, 1],
                color=colors[state % len(colors)],
                headwidth=5.0,
            )
            # ax[pair_id, state].set_title(
            #    "$A_{} x_{{t, {}{}}} + b_{} - x_t$".format(state + 1,
            #                                               cpair[0],
            #                                               cpair[1],
            #                                               state + 1)
            # )

            ax[pair_id, state].set_title(
                "$A_{} x_t + b_{} - x_t$".format(state + 1, state + 1)
            )
            # share x
            ax[pair_id, state].set_xlabel("$x_{{t, {} }}$".format(cpair[0]))
            # share y
            ax[pair_id, 0].set_ylabel("$x_{{t, {} }}$".format(cpair[1]))

    plt.tight_layout()

    if FIGURE_STORE:
        if fname is None:
            fname = "Vector Field Dynamics on Inputs.pdf"
        plt.savefig(os.path.join(OUTDIR, fname))
    else:
        plt.show()
    plt.close()

    return


def plot_vector_field_dynamics_datas(
    Aks,
    bks,
    datas,
    states,
    comb_obs=None,
    sharey=True,
    sharex=True,
    FIGURE_STORE=False,
    OUTDIR="",
    fname=None,
):
    # if you see an error check if you should be running datas instead
    if comb_obs is None:
        comb_obs = [(0, 1)]

    fig, ax = plt.subplots(
        len(comb_obs),
        len(Aks),
        figsize=(len(Aks) * 4, len(comb_obs) * 4),
        sharey=sharey,
        sharex=sharex,
    )

    if np.ndim(ax) == 1:
        ax = ax[None, :]

    for data_id, (data, dstates) in enumerate(zip(datas, states)):
        # State indices the columns
        print(' Plotting for data set {}'.format(data_id))
        for state, (Ak, bk) in enumerate(zip(Aks, bks)):
            # Select pair of coordinates
            x_t = data[dstates == state, :]

            # drop nan
            bad = np.isnan(x_t).any(1)
            x_t = x_t[~bad]

            if not np.any(x_t):
                print('No data with states {}'.format(state))
                continue

            for pair_id, cpair in enumerate(comb_obs):
                As = Ak[np.ix_(cpair, cpair)]
                bs = bk[np.ix_(cpair)]

                xy = x_t[:, cpair]
                dydt_m = xy.dot(As.T) + bs.T - xy

                ax[pair_id, state].quiver(
                    xy[:, 0],
                    xy[:, 1],
                    dydt_m[:, 0],
                    dydt_m[:, 1],
                    color=colors[state % len(colors)],
                    headwidth=5.0,
                )
                # ax[pair_id, state].set_title(
                #    "$A_{} x_{{t, {}{}}} + b_{} - x_t$".format(state + 1,
                #                                               cpair[0],
                #                                               cpair[1],
                #                                               state + 1)
                # )

                ax[pair_id, state].set_title(
                    "$A_{} x_t + b_{} - x_t$".format(state + 1, state + 1)
                )
                # share x
                ax[pair_id, state].set_xlabel("$x_{{t, {} }}$".format(cpair[0]))
                # share y
                ax[pair_id, 0].set_ylabel("$x_{{t, {} }}$".format(cpair[1]))

    plt.tight_layout()

    if FIGURE_STORE:
        if fname is None:
            fname = "Vector Field Dynamics on Inputs.pdf"
        plt.savefig(os.path.join(OUTDIR, fname))
    else:
        plt.show()
    plt.close()

    return


def plot_vector_field_dynamics_datas_formatted(
    Aks,
    bks,
    datas,
    states,
    num_states,
    num_cols=3,
    comb_obs=None,
    sharey=True,
    sharex=True,
    states_order=[],
    FIGURE_STORE=False,
    OUTDIR="",
    fname=None,
):
    # if you see an error check if you should be running datas instead
    if comb_obs is None:
        comb_obs = [(0, 1)]
    
    from math import ceil
    fig, axarr = plt.subplots(
        ceil(num_states/num_cols),
        num_cols,
        figsize=(9, 8),
        sharey=sharey,
        sharex=sharex,
    )

    if np.ndim(axarr) == 1:
        axarr = axarr[None, :]

    axarr = axarr.flatten()
    
    for ii, ax in enumerate(axarr):
        if ii >= num_states:
            ax.axis('off')
    
    if not np.any(states_order):
        states_order = np.arange(num_states)
    else:
        assert len(np.unique(states_order)) == num_states

    for data_id, (data, dstates) in enumerate(zip(datas, states)):
        # State indices the columns
        print(' Plotting for data set {}'.format(data_id))
        for state, ax in zip(states_order, axarr):
            
            Ak = Aks[state]
            bk = bks[state]
            
            print('Plotting state {}'.format(state))
            #ax = axarr[state]
            # Select pair of coordinates
            x_t = data[dstates == state, :]

            bad = np.isnan(x_t).any(1)
            x_t = x_t[~bad]

            if not np.any(x_t):
                print('No data with states {}'.format(state))
                continue

            for pair_id, cpair in enumerate(comb_obs):
                As = Ak[np.ix_(cpair, cpair)]
                bs = bk[np.ix_(cpair)]

                xy = x_t[:, cpair]
                dydt_m = xy.dot(As.T) + bs.T - xy

                ax.quiver(
                    xy[:, 0],
                    xy[:, 1],
                    dydt_m[:, 0],
                    dydt_m[:, 1],
                    color=colors[state % len(colors)],
                    headwidth=5.0,
                )
                
                # ax[pair_id, state].set_title(
                #    "$A_{} x_{{t, {}{}}} + b_{} - x_t$".format(state + 1,
                #                                               cpair[0],
                #                                               cpair[1],
                #                                               state + 1)
                # )
                
                ax.set_title('State z={}'.format(state))
                
                #if state % num_cols == 0:
                ax.set_xlabel("$x_{t,z}$")
                ax.set_ylabel("$y_{t,z}$")


    plt.tight_layout()

    if FIGURE_STORE:
        if fname is None:
            fname = "Vector Field Dynamics on Inputs.pdf"
        plt.savefig(os.path.join(OUTDIR, fname))
    else:
        plt.show()
    #plt.close()
    return