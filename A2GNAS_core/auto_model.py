from A2GNAS_core.search_space.search_space_config import SearchSpace
from A2GNAS_core.search_algorithm.search_algorithm import Search
from A2GNAS_core.model.logger import gnn_architecture_performance_load
from A2GNAS_core.model.fullsupervised_test import fullsupervised_scratch_train


class AutoModel(object):

    def __init__(self, graph_data, args):

        self.graph_data = graph_data
        self.args = args

        self.search_space = SearchSpace()

        self.search_algorithm = Search(self.search_space, self.args)

        self.search_model()

        self.derive_target_model()


    def search_model(self):

        self.search_algorithm.search_operator(self.graph_data)


    def derive_target_model(self):
        gnn_architecture_list, performance_list = gnn_architecture_performance_load(self.args.data_save_name)

        gnn_architecture_performance_dict = {}
        for gnn_architecture, value in zip(gnn_architecture_list, performance_list):
            gnn_architecture_performance_dict[str(gnn_architecture)] = value

        ranked_architecture_performance_dict = sorted(gnn_architecture_performance_dict.items(),
                                                      key=lambda x: x[1],
                                                      reverse = True)

        sorted_architecture_list = []
        top_k = self.args.return_top_k
        i = 0
        for key, value in ranked_architecture_performance_dict:
            if i == top_k:
                break
            else:
                sorted_architecture_list.append(eval(key))
                i += 1

        model_num = [num for num in range(len(sorted_architecture_list))]

        print(35 * "=" + " the testing start " + 35 * "=")

        for target_architecture, num in zip(sorted_architecture_list, model_num):
            print("test gnn architecture {}:\t".format(num + 1), str(target_architecture))

            ## train from scratch
            test_repeat = 5
            for i in range(test_repeat):
                valid_acc, test_acc, test_acc_std = fullsupervised_scratch_train(target_architecture, self.graph_data, args=self.args)
                print("{}-th run in all {} runs || Test_acc:{}±{}".format(i+1, test_repeat, test_acc, test_acc_std))

        print(35 * "=" + " the testing ending " + 35 * "=")
