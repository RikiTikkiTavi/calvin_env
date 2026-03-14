import itertools
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
import time
from typing import Union

import git
import numpy as np
import quaternion

# A logger for this file
logger = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class FpsController:
    def __init__(self, freq):
        self.loop_time = (1.0 / freq) * 10**9
        self.prev_time = time.time_ns()

    def step(self):
        current_time = time.time_ns()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            nano_sleep(self.loop_time - delta_t)
        self.prev_time = time.time_ns()


def xyzw_to_wxyz(arr):
    """
    Convert quaternions from pyBullet to numpy.
    """
    return [arr[3], arr[0], arr[1], arr[2]]


def wxyz_to_xyzw(arr):
    """
    Convert quaternions from numpy to pyBullet.
    """
    return [arr[1], arr[2], arr[3], arr[0]]


def nano_sleep(time_ns):
    """
    Spinlock style sleep function. Burns cpu power on purpose
    equivalent to time.sleep(time_ns / (10 ** 9)).

    Should be more precise, especially on Windows.

    Args:
        time_ns: time to sleep in ns

    Returns:

    """
    wait_until = time.time_ns() + time_ns
    while time.time_ns() < wait_until:
        pass


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between_quaternions(q1, q2):
    """
    Returns the minimum rotation angle between to orientations expressed as quaternions
    quaternions use X,Y,Z,W convention
    """
    q1 = xyzw_to_wxyz(q1)
    q2 = xyzw_to_wxyz(q2)
    q1 = quaternion.from_float_array(q1)
    q2 = quaternion.from_float_array(q2)

    theta = 2 * np.arcsin(np.linalg.norm((q1 * q2.conjugate()).vec))
    return theta


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


class EglDeviceNotFoundError(Exception):
    """Raised when EGL device cannot be found"""


def get_egl_device_id(cuda_id: int) -> Union[int]:
    """Map a virtual CUDA device index to the corresponding EGL device index.

    SLURM (and other launchers) set ``CUDA_VISIBLE_DEVICES`` to restrict which
    physical GPUs a process can see, remapping them to virtual indices starting
    at 0.  ``EGL_CUDA_DEVICE_NV`` in ``EGL_options.o`` returns the *physical*
    CUDA device index, so comparing it directly to the virtual ``cuda_id``
    gives a wrong match when ``CUDA_VISIBLE_DEVICES`` is set.

    This function translates ``cuda_id`` (virtual) to the physical device index
    before probing, so that the comparison is always apples-to-apples.

    >>> i = get_egl_device_id(0)
    >>> isinstance(i, int)
    True
    """
    assert isinstance(cuda_id, int), "cuda_id has to be integer"

    # Resolve virtual cuda_id → physical device index.
    # CUDA_VISIBLE_DEVICES="2,3" means virtual 0 → physical 2, virtual 1 → physical 3.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        physical_ids = [int(x) for x in cuda_visible.split(",")]
        assert cuda_id < len(physical_ids), (
            f"cuda_id={cuda_id} out of range for CUDA_VISIBLE_DEVICES={cuda_visible!r}"
        )
        physical_cuda_id = physical_ids[cuda_id]
    else:
        physical_cuda_id = cuda_id

    dir_path = Path(__file__).absolute().parents[2] / "egl_check"
    if not os.path.isfile(dir_path / "EGL_options.o"):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Building EGL_options.o")
            subprocess.call(["bash", "build.sh"], cwd=dir_path)
        else:
            # In case EGL_options.o has to be built and multiprocessing is used, give rank 0 process time to build
            time.sleep(5)
    result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path)
    print(result)
    # EGL device choice: 0 of 1 (from EGL_VISIBLE_DEVICE)
    n = int(re.search(r"of\s+([0-9]+)", result.stderr.decode("utf-8")).group(1))
    for egl_id in range(n):
        my_env = os.environ.copy()
        my_env["EGL_VISIBLE_DEVICE"] = str(egl_id)
        result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path, env=my_env)
        match = re.search(r"CUDA_DEVICE=[0-9]+", result.stdout.decode("utf-8"))
        if match:
            current_cuda_id = int(match[0].split("=")[1])
            # Compare physical indices: EGL_CUDA_DEVICE_NV returns physical device ID.
            if physical_cuda_id == current_cuda_id:
                return egl_id
    raise EglDeviceNotFoundError


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def set_egl_device(device):
    assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
    try:
        cuda_id = device.index if device.type == "cuda" else 0
    except AttributeError:
        cuda_id = device
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        logger.warning(
            "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
            "When using DDP with many GPUs this can lead to OOM errors. "
            "Did you install PyBullet correctly? Please refer to VREnv README"
        )
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")


def count_frames(directory):
    """
    counts the number of consecutive pickled frames in directory

    Args:
        directory: str of directory

    Returns:
         0 for none, otherwise >0
    """

    for i in itertools.count(start=0):
        pickle_file = os.path.join(directory, f"{str(i).zfill(12)}.pickle")
        if not os.path.isfile(pickle_file):
            return i


def get_episode_lengths(load_dir, num_frames):
    episode_lengths = []
    render_start_end_ids = [[0]]
    i = 0
    for frame in range(num_frames):
        file_path = os.path.abspath(os.path.join(load_dir, f"{str(frame).zfill(12)}.pickle"))
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            done = data["done"]
            if not done:
                i += 1
            else:
                episode_lengths.append(i)
                render_start_end_ids[-1].append(frame + 1)
                render_start_end_ids.append([frame + 1])
                i = 0
    render_start_end_ids = render_start_end_ids[:-1]
    return episode_lengths, render_start_end_ids


if __name__ == "__main__":
    import doctest

    doctest.testmod()
