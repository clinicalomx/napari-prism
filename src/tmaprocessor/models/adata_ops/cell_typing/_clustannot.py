from _helpers import ObsAnnotator

class AnnDataAnnotator():
    """ Annotates AnnData.obs columns based on mappings, conditions and subsets. """

    def __init__(self, adata):
        self.adata = adata

