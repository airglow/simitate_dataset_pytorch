"""
File: simitate_dataset.py
Author: Raphael Memmesheimer
Email: raphael@uni-koblenz.de
Github: https://github.com/airglow
Description: Simitate interface for pytorch dataset integration

"""

# from torch.utils.data import Dataset
from os.path import join
# import pandas as pd
import collections
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive,\
    check_integrity, list_dir, list_files
from simitate.trajectory_loader import SimitateTrajectoryLoader
# from trajectory_loader import plot_trajectory

def plot_trajectory(subplot, label, trajectory_data, char=None):
    # if frame == "baseline_trajectory":
        # subplot.plot(trajectory_data[:, 1],
                     # trajectory_data[:, 2],
                     # trajectory_data[:, 3], "ro", label=label)
    # else:
        subplot.plot(trajectory_data[:, 0],
                     trajectory_data[:, 1],
                     trajectory_data[:, 2], label=label)



class SimitateTrajectoriesDataset(VisionDataset):
    """
    Simitate Dataset class, handles dowloading, integrity test and creates a
    usable data structure.


    Folder structure is like
    | .
    |---> category (currenlty: ['motions', 'basic_motions', 'complex', 'sequential'])
    |-----> class  [...]
    |------> person
    |-------> <sequence_name>.csv
    """
    folder = 'simitate_trajectories'
    download_url_prefix = 'https://agas.uni-koblenz.de/simitate/data/simitate/data/'
    filename = "simitate_trajectories"
    zips_md5 = {
        # TODO update this, when updating zip
        'simitate_trajectories': 'c499ff41e84a30b0ca21d0b89750fc85',
        'simitate_sequences': ""}

    categories = []
    classes = []
    persons = []
    trajectories = []
    trajectory_data = []
    trajectory_loader = SimitateTrajectoryLoader()
    trajectory_class = []

    def __init__(self, root, download=False, classes=None, categories=None):

        """Handles the dataset initialization

        :root: string where to search or download the dataset.
        :download: if `true` the dataset will be downloaded when not already
                   done.
        :classes:  a classes filter, if not empty then only the listed classes\
                   will be loaded
        :categories: a category filter, if not empty only listed categories\
                     will be taken into account
        """
        super(SimitateTrajectoriesDataset, self).__init__(join(root,
                                                               self.folder))
        self.target_folder = self.root

        self.categories = []
        self.classes = []
        self.persons = []
        self.trajectories = []
        self.trajectory_data = []
        self.trajectory_loader = SimitateTrajectoryLoader()
        self.trajectory_class = []

        if download:
            self.download()

        self.categories = list_dir(self.target_folder)
        for category in self.categories:
            if categories:  # filter categories
                if category not in categories:
                    continue
            category_folder = join(self.target_folder, category)
            classes_in_category = list_dir(category_folder)
            self.classes.extend(classes_in_category)

            for current_class in classes_in_category:
                if classes:  # filter classes
                    if current_class not in classes:
                        continue

                class_folder = join(category_folder, current_class)
                persons_in_class = list_dir(class_folder)
                self.persons.extend(persons_in_class)
                for person in persons_in_class:
                    person_folder = join(class_folder, person)
                    sequences_from_person = list_dir(person_folder)
                    for sequence in sequences_from_person:
                        sequence_folder = join(person_folder, sequence)
                        trajectories = list_files(sequence_folder,
                                                  suffix=".csv")
                        #  there can be only one trajectory per sequence
                        # print(trajectories)
                        if not len(trajectories) == 1:
                            print ("There are at least two potential\
                                    trajectory files in here.\
                                    Take a closer look at this. %s %s" %
                                    (sequence_folder, trajectories))
                            continue

                        if len(trajectories) == 0:
                            print ("There is no trajectory file in here.\
                                    Take a closer look at this. %s %s" %
                                    (sequence_folder, trajectories))
                            continue

                        self.trajectories.extend(trajectories)
                        try:
                            trajectory_file = join(sequence_folder, trajectories[0])
                            # print("Trajectory File", trajectory_file)
                            if trajectory_file[0] == "openpose.csv":
                                continue
                            # # using panda
                            #  trajectory_data = pd.read_csv(trajectory_file)
                            # #  using simitate trajectory loader

                            # print ("Load", trajectory_file)
                            self.trajectory_loader.load_trajectories(trajectory_file)
                            #self.trajectory_loader.load_trajectories(trajectory_file, ["hand"])
                            
                            current_trajectory = self.trajectory_loader.trajectories["hand"][:,1:] # first column is time
                            current_trajectories = self.trajectory_loader.trajectories
                            #traj = traj[:][:][:][1:]
                            #print("traj1", traj)
                            current_trajectory_plot = None  #  self.trajectory_loader.plot_trajectories(show=False)
                            trajectory_data = {"class_id": self.classes.index(current_class),
                                              "trajectory": torch.from_numpy(current_trajectory).float(),
                                              "trajectories": current_trajectories,
                                              "trajectory_file": trajectory_file,
                                              "plot": current_trajectory_plot}
                            # print(type(trajectory_data))
                            # print(trajectory_data["trajectory"])
                            self.trajectory_data.append(trajectory_data)
                        except Exception as e:
                            print(sequence_folder, trajectories)
                            print(e)
                            #raise(e)

        print ("Categories: %s\nClasses: %s\nPerons: %s\nTrajectories: %s\n" %
               (self.categories, self.classes, self.persons, self.trajectories))

        print ("Counts:\nCategories: %d\nClasses: %d\nPerons: %d\nTrajectories: %d\n" %
               (len(self.categories), len(self.classes), len(self.persons), len(self.trajectories)))

        print([(item, count) for item, count in collections.Counter(self.trajectories).items() if count > 1])

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, index):
        """returns item
        :returns: trajectory and class id. Note the trajectory starts at 1, as
                  the first element is the time

        """
        # TODO: get image sequence here
        return self.trajectory_data[index]["trajectory"], self.trajectory_data[index]["class_id"]

    def plot(self, index, usetex=False):
        fig = plt.figure()

        plt.rc('text', usetex=usetex)
        plt.rc('font', family='serif')
        ax = fig.add_subplot("111", projection='3d')
        ax.set_title(self.trajectories[index].replace("_", "\_"))
        plot_trajectory(subplot=ax, label="trajectory",
                            trajectory_data=self.trajectory_data[index]["trajectory"].numpy())
        ax.legend()
        # if show:
        plt.show()
        return fig


    def _check_integrity(self):
        if not check_integrity(join(self.root, self.filename + '.zip'),
                               self.zips_md5[self.filename]):
            return False
        return True

    def download(self):
        """Downloads the simitate dataset and extracts it. It also checks if
        the dataset was already updated and checks integrity.

        """
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # filename = self._get_target_folder()
        zip_filename = self.filename + ".zip"
        url = self.download_url_prefix + '/' + zip_filename
        download_and_extract_archive(url, self.root,
                                     filename=zip_filename,
                                     md5=self.zips_md5[self.filename])
