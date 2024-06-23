import sys
sys.path.append('./case_study/Thread')
import numpy as np

class ThreadCoarseningInterface(object):
    """
    A model for predicting OpenCL thread coarsening factors.

    Attributes
    ----------
    __name__ : str
        Model name
    __basename__ : str
        Shortened name, used for files
    """
    __name__ = None
    __basename__ = None

    def init(self, args) -> None:
        """
        Initialize the model.

        Do whatever is required to setup a new thread coarsening model here.
        This method is called prior to training and predicting.
        This method may be omitted if no initial setup is required.

        Parameters
        ----------
        seed : int
            The seed value used to reproducible results. May be 'None',
            indicating that no seed is to be used.
        """
        pass

    def save(self, outpath: str) -> None:
        """
        Save model state.

        This must capture all of the relevant state of the model. It is up
        to implementing classes to determine how best to save the model.

        Parameters
        ----------
        outpath : str
            The path to save the model state to.
        """
        raise NotImplementedError

    def fit(self, seed: int):
        raise NotImplementedError

    def restore(self, inpath: str) -> None:
        """
        Load a trained model from file.

        This is called in place of init() if a saved model file exists. It
        must restore all of the required model state.

        Parameters
        ----------
        inpath : str
            The path to load the model from. This is the same path as
            was passed to save() to create the file.
        """
        raise NotImplementedError

    def train(self,
              sequences: np.array, y_1hot: np.array, verbose: bool = False) -> None:
        """
        Train a model.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        cascading_y : np.array
            An array of classification labels of shape(n,7). Used for the cascading
            model.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        y_1hot : np.array
            An array of optimal coarsening factors of shape (n,6), in 1-hot encoding.

        verbose: bool, optional
            Whether to print verbose status messages during training.
        """
        raise NotImplementedError

    def predict(self, sequences: np.array) -> np.array:
        """
        Make predictions for programs.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        Returns
        -------
        np.array
            Predicted 'y' values (optimal thread coarsening factors) with shape (n,1).
        """
        raise NotImplementedError

    def predict_proba(self, sequences: np.array):
        raise NotImplementedError

    def predict_model(self, sequences: np.array):
        raise NotImplementedError

    def data_partitioning(self, dataset, calibration_ratio=0.1):
        pass

    def feature_extraction(self, X):
        pass


# demo split
# Thread_dataset = r'../benchmarks/Thread'
#
# for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
#     platform_name = platform2str(platform)
#     # load data
#     oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
#     y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
#     y_1hot = get_onehot(oracles, platform)
#     X_cc, y_cc = get_magni_features(df, oracles, platform)
#
#
# ThreadCoarseningInterface.data_partitioning(dataset="", calibration_ratio=0.1)