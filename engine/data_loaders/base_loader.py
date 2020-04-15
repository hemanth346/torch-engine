import abc

class BaseLoader(abc.ABC):
    mean = (0, )
    std = (0, )
    classes = []

    @abc.abstractmethod
    # @staticmethod
    def get_loaders(batch_size:int=64, augmentations=[], seed:int=None, shuffle=True, num_workers=4, pin_memory=True):
        pass



    def show_images(self, number=5, ncols=5, figsize=(15, 10), iterate=True):

        if iterate:# or not images:
            self.__images, self.__labels = next(iter(self.loader))
            # images, labels = self.__images, self.__labels

        images, labels = self.__images, self.__labels
        # selecting random sample of number
        img_list = random.sample(range(1, images.shape[0]), number)

        self.set_plt_param()
        rows = (number//ncols) + 1
        axes = self.make_grid(rows, ncols, figsize)
        for idx, label in enumerate(img_list):
            img = unnormalize_chw_image(images[label], self.mean, self.std)
            axes[idx].imshow(img, interpolation='bilinear')
            axes[idx].set_title(self.targets[labels[label]])
        # Hide empty subplot boundaries
        [ax.set_visible(False) for ax in axes[idx + 1:]]
        plt.show()

    @staticmethod
    def make_grid(nrows, ncols=3, figsize=(6.0, 4.0)):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        axes = ax.flatten()
        return axes
        pass

    @staticmethod
    def set_plt_param():
        rc = {"axes.spines.left": False,
              "axes.spines.right": False,
              "axes.spines.bottom": False,
              "axes.spines.top": False,
              "axes.grid": False,
              "xtick.bottom": False,
              "xtick.labelbottom": False,
              "ytick.labelleft": False,
              "ytick.left": False}
        plt.rcParams.update(rc)
