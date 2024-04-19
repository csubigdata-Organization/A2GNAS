class SearchSpace(object):
    """
    Loading the search space dict
    """
    def __init__(self):
        ### PM: Propagation Mechanisms, PS: Propagation Steps, AS: Asymmetric Strength
        self.stack_gnn_architecture = ['augmentation'] + \
                                      ['PM', 'local_pooling', 'global_pooling', 'PS', 'AS']

        self.space_dict = {
            'augmentation': ['attribute_masking', 'edge_perturbation', 'node_dropping', 'autogcl'],
            'PM': ['GCN_PM', 'GAT_PM', 'SAGE_PM', 'GIN_PM', 'Graph_PM', 'General_PM'],
            'local_pooling': ['TopKPool', 'SAGPool', 'ASAPool', 'PANPool', 'HopPool', 'GCPool', 'GAPPool', 'None'],
            'global_pooling': ['global_max', 'global_mean', 'global_add'],
            'PS': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'AS': ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        }
