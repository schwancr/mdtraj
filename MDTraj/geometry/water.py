
import numpy as np
import mdtraj as md

def compute_OO_distances(traj, which='all'):
    """
    Compute the O-O distances in a trajectory for each water molecule
    in the dataset.

    Parameters:
    -----------
    traj : mdtraj.trajectory
        trajectory with water molecules in it
    which : str, optional
        which distances to return:
            - 'all' : compute and return all distances
            - 'firstshell' : compute all distances but return only
                distances in the first solvation shell
            - 'secondshell' : compute all distances but return only
                distances in the second shell
            - 'bothshells' : compute all distances but return only distances
                to the first and second solvation shells of each water
                molecule
            
    Returns:
    --------
    distances : np.ndarray
        array with shape (n_timepoints, n_waters, n_distances)
    """
    possible_which = ['all', 'firstshell', 'secondshell', 'bothshells']

    which = which.lower()
    if not which in possible_which:
        raise Exception("which must be one of %s" % str(possible_which))

    shells = False
    if 'shell' in which:
        shells = True

    oxygens = np.array([a.index for a in traj.top.atoms if (a.name == 'O') and (a.residue.name in ['HOH', 'SOL'])])
    n_waters = len(oxygens)

    water_inds = np.array([(i, j) for i in xrange(n_waters) for j in xrange(i + 1, n_waters)])
    atom_pairs = oxygens[water_inds]

    distances = md.compute_distances(traj, atom_pairs)
    distances = md.geometry.contact.squareform(distances, water_inds)

    n_frames, n_waters, _ = distances.shape

    if not shells:
        distances.sort() # sort the last axis
        distances = distances[:, :, 1:] # remove the self-distance

    else:
        first_shell_inds = np.argsort(distances, axis=2)[:, :, 1:5] 
        # first shell contains the four closest waters
        
        if which == 'firstshell':
            inds = first_shell_inds
        else:
            i0 = np.ones((n_frames, n_waters, 1), dtype=int) * np.arange(n_frames).reshape((-1, 1, 1))
            second_shell_inds = first_shell_inds[i0, first_shell_inds].reshape((n_frames, n_waters, -1))
            # this array contains the first shell of the first shell waters, which is the second shell
            # of the original water. 
            # the reshape changes the shape from (n_frames, n_waters, 4, 4) -> (n_frames, n_waters, 16)
            if which == 'secondshell':
                inds = second_shell_inds

            else: # we want both shells
                inds = np.concatenate([first_shell_inds, second_shell_inds], axis=2)
                # add the first shell to this second shell and we get both the first and second shell

        i0 = np.ones((n_frames, n_waters, 1), dtype=np.int) * np.arange(n_frames).reshape((-1, 1, 1))
        i1 = np.ones((n_frames, n_waters, 1), dtype=np.int) * np.arange(n_waters).reshape((1, -1, 1))
        distances = distances[i0, i1, inds]
        # this contains the (sometimes redundant) distances from the original water to the waters in
        # the first and/or second shell
        
    return distances
