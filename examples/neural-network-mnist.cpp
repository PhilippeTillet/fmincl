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

typedef umintl::backend::eigen_types<ScalarType> BackendType;

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

MatrixType read_mnist_labels(std::string  filename)
{
  MatrixType Y;
  std::ifstream fs(filename.c_str(), std::ios::binary);
  if(fs) {
    int magic_number, num_images;
    fs.read((char*)&magic_number, sizeof(magic_number));
    fs.read((char*)&num_images, sizeof(num_images));
    if (magic_number != 2049) {
      swap(magic_number);
      swap(num_images);
    }

    Y = MatrixType::Zero(10, num_images);

    for (int i=0; i<num_images; ++i) {
      unsigned char temp=0;
      fs.read((char*)&temp,sizeof(temp));
      Y((int) temp, i) = 1.0;
    }
    fs.close();
  } else {
    std::cerr << "error reading file: " << filename << std::endl;
    exit(1);
  }
  return Y;
}

class neural_net{
public:
    neural_net(MatrixType const & training_data, MatrixType const & training_labels,
       MatrixType const & cross_validation_data, MatrixType cross_validation_labels,
       std::size_t n_hidden, ScalarType lambda=0.01) :
        training_data_(training_data), training_labels_(training_labels)
        ,cross_validation_data_(cross_validation_data), cross_validation_labels_(cross_validation_labels)
      ,n_in_(training_data.rows()), n_hidden_(n_hidden), n_out_(training_labels.rows()), lambda_(lambda)
    {
        weights_1_.resize(n_hidden_,n_in_);
        bias_1_.resize(n_hidden_);
        weights_2_.resize(n_out_,n_hidden_);
        bias_2_.resize(n_out_);

    }

    std::size_t n_params() const{
        return n_hidden_*n_in_+n_hidden_ + n_out_*n_hidden_+n_out_;
    }
    void operator()(VectorType const & X, ScalarType * val, VectorType * grad)const{

        //Unroll
        std::size_t offset = 0;
        //Layer1
        for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
            for(std::size_t j = 0 ; j < n_in_ ; ++j)
                weights_1_(i,j) = X[offset++];
        for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
            bias_1_(i) = X[offset++];
        //Layer2
        for(std::size_t i = 0 ; i < n_out_ ; ++i)
            for(std::size_t j = 0 ; j < n_hidden_ ; ++j)
                weights_2_(i,j) = X[offset++];
        for(std::size_t i = 0 ; i < n_out_ ; ++i)
            bias_2_(i) = X[offset++];

        //Hidden
        Z1 = weights_1_*training_data_;
        Z1.colwise() += bias_1_;
        A1 = -Z1;
        A1 = 1/(1+A1.array().exp());

        //Final output
        Z2 = weights_2_*A1;
        Z2.colwise() += bias_2_;
        A2 = -Z2;
        A2 = 1/(1+A2.array().exp());

        D2 = (A2 - training_labels_);
        if(val){
            *val = 0.5*D2.array().pow(2).sum() + 0.5*lambda_*(weights_1_.array().pow(2).sum()+weights_2_.array().pow(2).sum());
        }
        D2 = D2.array()*A2.array()*(1-A2.array());

        if(grad){
            dweights_2_ = D2*A1.transpose() + lambda_*weights_2_;

            dbias_2_ = D2.rowwise().sum();

            D1 = weights_2_.transpose()*D2;
            D1 = D1.array()*A1.array()*(1-A1.array());
            dweights_1_ = D1*training_data_.transpose() + lambda_*weights_1_;
            dbias_1_ = D1.rowwise().sum();



            //Reroll
            std::size_t offset = 0;
            //Layer1
            for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
                for(std::size_t j = 0 ; j < n_in_ ; ++j)
                    (*grad)[offset++] = dweights_1_(i,j);
            for(std::size_t i = 0 ; i < n_hidden_ ; ++i)
                (*grad)[offset++] = dbias_1_(i);

            //Layer2
            for(std::size_t i = 0 ; i < n_out_ ; ++i)
                for(std::size_t j = 0 ; j < n_hidden_ ; ++j)
                    (*grad)[offset++] = dweights_2_(i,j);
            for(std::size_t i = 0 ; i < n_out_ ; ++i)
                (*grad)[offset++] = dbias_2_(i);
        }


    }


private:
    mutable MatrixType weights_1_;
    mutable VectorType bias_1_;

    mutable MatrixType weights_2_;
    mutable VectorType bias_2_;

    MatrixType const & training_data_;
    MatrixType const & training_labels_;
    MatrixType const & cross_validation_data_;
    MatrixType const & cross_validation_labels_;

    std::size_t n_in_;
    std::size_t n_hidden_;
    std::size_t n_out_;

    mutable MatrixType Z1;
    mutable MatrixType A1;
    mutable MatrixType Z2;
    mutable MatrixType A2;

    mutable MatrixType D2;
    mutable MatrixType D1;

    mutable MatrixType dweights_1_;
    mutable MatrixType dbias_1_;

    mutable MatrixType dweights_2_;
    mutable MatrixType dbias_2_;

    ScalarType lambda_;
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
    MatrixType training_label = read_mnist_labels(path + "/train-labels.idx1-ubyte");
    MatrixType testing_data = read_mnist_images(path + "/t10k-images.idx3-ubyte");
    MatrixType testing_label = read_mnist_labels(path + "/t10k-labels.idx1-ubyte");

//    MatrixType training_data = MatrixType::Random(10,100);
//    MatrixType training_label = MatrixType::Random(10,100);
//    MatrixType testing_data = MatrixType::Random(10,100);
//    MatrixType testing_label = MatrixType::Random(10,100);

    std::cout << "done!" << std::endl;



    std::cout << "#Initializing the network..." << std::flush;
    neural_net network(training_data,training_label,testing_data,testing_label,500);
    VectorType W0(network.n_params());
    VectorType Res(network.n_params());
    for(std::size_t i = 0 ; i < W0.rows() ; ++i)
        W0(i) = (ScalarType)rand()/RAND_MAX;

    std::cout << "done!" << std::endl;

    //std::cout << "#Checking gradient..." << std::flush;
    //std::cout << "Maximum relative error : " << umintl::check_grad<BackendType>(network,W0,W0.rows(),1e-6) << std::endl;

    umintl::minimizer<BackendType> optimization;
    optimization.direction = new umintl::conjugate_gradient<BackendType>();
    //optimization.direction = new umintl::steepest_descent<BackendType>();
    //optimization.direction = new umintl::quasi_newton<BackendType>(new umintl::lbfgs<BackendType>(8));
    optimization.verbosity_level=2;
    optimization(Res,network,W0,W0.rows());




}
