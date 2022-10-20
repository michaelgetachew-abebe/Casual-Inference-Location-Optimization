from copyreg import pickle
import pandas as pd
import dvc.api as dvc
import io

class Util:
    def __init__(self) -> None:
        pass

    def load_from_dvc(self, path, repo, rev, low_memory=True):
        """
        Loads the data from dvc
        """
        try:
            data = dvc.read(path=path,repo=repo, rev=rev)
            df = pd.read_csv(io.StringIO(data),low_memory=low_memory)

            return df
        except Exception as e:
            print("Something went wrong!",e)