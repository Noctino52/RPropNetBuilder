from learning import batch_learning
import utility as utl
from mnist import MNIST
from net import MultilayerNet


def run():
    # parametri di default 
    n_hidden_layers = 2
    n_hidden_nodes_per_layer = [35,35]
    act_fun_codes = [0,0,1]
    error_fun_code = 1

    # caricamento dataset
    mndata = MNIST('./data/')
    X, t = mndata.load_training()
    X = utl.get_mnist_data(X)
    t = utl.get_mnist_labels(t)

    X, t = utl.get_random_dataset(X, t, n_samples = 10000)
    X = utl.get_scaled_data(X)

    X_train, X_test, t_train, t_test = utl.train_test_split(X, t, test_size = 0.25)
    X_train, X_val, t_train, t_val = utl.train_test_split(X_train, t_train, test_size = 0.3334)

    net = MultilayerNet(n_hidden_layers= n_hidden_layers, n_hidden_nodes_per_layer= n_hidden_nodes_per_layer, 
                        act_fun_codes= act_fun_codes, error_fun_code= error_fun_code)
    
    net = batch_learning(net, X_train, t_train, X_val, t_val)
    y_test = net.sim(X_test)
    utl.print_result(y_test,t_test)

run()
