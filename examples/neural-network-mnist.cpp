#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "umintl/check_grad.hpp"
#include "umintl/backends/eigen.hpp"
#include "umintl/minimize.hpp"

typedef double ScalarType;
typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> VectorType;
typedef Eigen::Matrix<int,Eigen::Dynamic,1> LabelType;

typedef umintl::backend::eigen_types<ScalarType> BackendType;

inline void sigmoid(MatrixType const & in, MatrixType & out)
{
    out = -in;
    out = 1/(1+out.array().exp());
}

inline void swap(int &val)
{
        val = (val<<24) | ((val<<8) & 0x00ff0000) | ((val>>8) & 0x0000ff00) | (val>>24);
}

MatrixType read_mnist_images(std::string filename)
{
  MatrixType X;
  std::ifstream fs(filename.c_str(), std::ios::binary);
  if(fs) {
    int magic_number, num_images, num_rows, num_columns;
    fs.read((char*)&magic_number, sizeof(magic_number));
    fs.read((char*)&num_images, sizeof(num_images));
    fs.read((char*)&num_rows, sizeof(num_rows));
    fs.read((char*)&num_columns, sizeof(num_columns));
    if (magic_number != 2051) {
      swap(magic_number);
      swap(num_images);
      swap(num_rows);
      swap(num_columns);
    }
    //num_images = 1000;
    X = MatrixType::Zero(num_rows*num_columns, num_images);

    for (int i=0; i<num_images; ++i) {
      for (int j=0; j<num_rows*num_columns; ++j) {
        unsigned char temp=0;
        fs.read((char*)&temp,sizeof(temp));
        X(j,i) = (double) temp;
      }
    }
    fs.close();
  } else {
    std::cerr << "error reading file: " << filename << std::endl;
    exit(1);
  }
  return X;
}

LabelType read_mnist_labels(std::string  filename)
{
  LabelType Y;
  std::ifstream fs(filename.c_str(), std::ios::binary);
  if(fs) {
    int magic_number, num_images;
    fs.read((char*)&magic_number, sizeof(magic_number));
    fs.read((char*)&num_images, sizeof(num_images));
    if (magic_number != 2049) {
      swap(magic_number);
      swap(num_images);
    }
    //num_images = 1000;

    Y = LabelType::Zero(num_images);

    for (int i=0; i<num_images; ++i) {
      unsigned char temp=0;
      fs.read((char*)&temp,sizeof(temp));
      Y(i) = (int) temp;
    }
    fs.close();
  } else {
    std::cerr << "error reading file: " << filename << std::endl;
    exit(1);
  }
  return Y;
}


struct temporaries_holder{

};



class neural_net{
private:
    void unroll(VectorType const & X) const{
        std::size_t offset = 0;
        //Layer1
        for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
            for(std::size_t j = 0 ; j < n_in_ ; ++j)
                weights_1(i,j) = X[offset++];
        for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
            bias_1(i) = X[offset++];

        //Layer2
        for(std::size_t i = 0 ; i < n_out_ ; ++i)
            for(std::size_t j = 0 ; j < n_hidden_ ; ++j)
                weights_2(i,j) = X[offset++];
        for(std::size_t i = 0 ; i < n_out_ ; ++i)
            bias_2(i) = X[offset++];

    }

    void feedforward(MatrixType const & data) const{
        //Hidden
        Z1 = weights_1*data;
        Z1.colwise() += bias_1;
        sigmoid(Z1,A1);

        //Final output
        Z2 = weights_2*A1;
        Z2.colwise() += bias_2;
        A2 = Z2.array().exp();
        VectorType sum_exp = A2.colwise().sum();
        for(std::size_t i = 0 ; i < A2.rows() ; ++i)
            A2.row(i) = A2.row(i).array() / sum_exp.transpose().array();
    }

    void backpropagate(LabelType labels) const{
        D2 = A2;
        for(std::size_t j = 0 ; j < D2.cols() ; ++j){
            D2(labels(j),j)-=1;
        }
        dbias_2 = D2.rowwise().sum();
        dweights_2 = D2*A1.transpose();

        D1 = weights_2.transpose()*D2;
        D1 = D1.array()*A1.array()*(1-A1.array());
        dweights_1 = D1*training_data_.transpose();
        dbias_1 = D1.rowwise().sum();
    }

    ScalarType get_cost(LabelType const & labels) const{
        ScalarType cross_entropy = 0;
        for(std::size_t j = 0 ; j < Z2.cols() ; ++j){
            cross_entropy-=log(A2(labels(j),j));
        }
        return cross_entropy;
    }

public:
    class early_stopper : public umintl::stopping_criterion<BackendType>{
    public:
        early_stopper(neural_net const & net, MatrixType const & validation_data, LabelType const & validation_labels) : net_(net), validation_data_(validation_data), validation_labels_(validation_labels){ }
        bool operator()(umintl::detail::optimization_context<BackendType> & c){
            //Unroll
            net_.unroll(c.x());
            net_.feedforward(validation_data_);
            ScalarType cross_entropy = net_.get_cost(validation_labels_);
            std::cout << cross_entropy << std::endl;
            return false;
        }

    private:
        neural_net const & net_;

        MatrixType const & validation_data_;
        LabelType const & validation_labels_;
    };

    early_stopper * create_early_stopping(MatrixType const & validation_data, LabelType const & validation_labels)
    {
        return new early_stopper(*this,validation_data,validation_labels);
    }


public:
    neural_net(MatrixType const & training_data, LabelType const & training_labels,
       std::size_t n_hidden) :
         training_data_(training_data), training_labels_(training_labels)
      ,n_in_(training_data.rows()), n_hidden_(n_hidden), n_out_(training_labels.maxCoeff()+1)
    {
        weights_1.resize(n_hidden_,n_in_);
        bias_1.resize(n_hidden_);
        weights_2.resize(n_out_,n_hidden_);
        bias_2.resize(n_out_);
    }

    std::size_t n_params() const{
        return n_hidden_*n_in_+n_hidden_ + n_out_*n_hidden_+n_out_;
    }

    void operator()(VectorType const & X, ScalarType * val, VectorType * grad)const{
        //Unroll
        unroll(X);
        //Hidden
        feedforward(training_data_);
        if(val){
            *val = get_cost(training_labels_);
        }
        if(grad){
            backpropagate(training_labels_);
            //Reroll
            std::size_t offset = 0;
            //Layer1
            for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
                for(std::size_t j = 0 ; j < n_in_ ; ++j)
                    (*grad)[offset++] = dweights_1(i,j);
            for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
                (*grad)[offset++] = dbias_1(i);

            //Layer2
            for(std::size_t i = 0 ; i < n_out_ ; ++i)
                for(std::size_t j = 0 ; j < n_hidden_ ; ++j)
                    (*grad)[offset++] = dweights_2(i,j);
            for(std::size_t i = 0 ; i < n_out_ ; ++i)
                (*grad)[offset++] = dbias_2(i);
        }
    }

private:
    friend class early_stopper;

    mutable MatrixType weights_1;
    mutable VectorType bias_1;

    mutable  MatrixType weights_2;
    mutable VectorType bias_2;

    mutable MatrixType Z1;
    mutable MatrixType A1;
    mutable MatrixType Z2;
    mutable MatrixType A2;

    mutable MatrixType D2;
    mutable MatrixType D1;

    mutable MatrixType dweights_1;
    mutable MatrixType dbias_1;

    mutable MatrixType dweights_2;
    mutable MatrixType dbias_2;

    MatrixType const & training_data_;
    LabelType const & training_labels_;

    std::size_t n_in_;
    std::size_t n_hidden_;
    std::size_t n_out_;
};

int main(int argc, char* argv[]){
    if (argc != 2) {
      std::cout << "please provide path to mnist data ..." << std::endl;
      std::cout << "you can download the dataset at http://yann.lecun.com/exdb/mnist/" << std::endl;
      std::cout << std::endl << "usage: " << argv[0] << " path_to_data" << std::endl << std::endl;
      return 1;
    }

    std::string path = argv[1];

    std::cout << "#Reading data..." << std::flush;
    MatrixType training_data = read_mnist_images(path + "/train-images.idx3-ubyte");
    LabelType training_label = read_mnist_labels(path + "/train-labels.idx1-ubyte");
    MatrixType testing_data = read_mnist_images(path + "/t10k-images.idx3-ubyte");
    LabelType testing_label = read_mnist_labels(path + "/t10k-labels.idx1-ubyte");

//    MatrixType training_data = MatrixType::Random(5,10);
//    LabelType training_label = LabelType::Zero(10);
//    for(std::size_t i = 0 ; i < 10 ; ++i)
//        training_label(i) = rand()%10;
//    MatrixType testing_data = MatrixType::Random(5,10);
//    LabelType testing_label = LabelType::Zero(10);
//    for(std::size_t i = 0 ; i < 10 ; ++i)
//        testing_label(i) = rand()%10;

    std::cout << "done!" << std::endl;



    std::cout << "#Initializing the network..." << std::flush;

    neural_net network(training_data,training_label,700);

    VectorType W0(network.n_params());
    VectorType Res(network.n_params());
    for(std::size_t i = 0 ; i < W0.rows() ; ++i)
        W0(i) = (ScalarType)rand()/RAND_MAX - 0.5;

    std::cout << "done!" << std::endl;

    //std::cout << "#Checking gradient..." << std::flush;
    //std::cout << "Maximum relative error : " << umintl::check_grad<BackendType>(network,W0,W0.rows(),1e-6) << std::endl;

    umintl::minimizer<BackendType> optimization;
    //optimization.direction = new umintl::conjugate_gradient<BackendType>();
    //optimization.direction = new umintl::steepest_descent<BackendType>();
    optimization.direction = new umintl::quasi_newton<BackendType>(new umintl::lbfgs<BackendType>(8));
    optimization.stopping_criterion = network.create_early_stopping(testing_data,testing_label);
    optimization.verbosity_level=2;
    optimization(Res,network,W0,W0.rows());




}
