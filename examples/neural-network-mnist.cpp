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

inline void softmax(MatrixType const & in, MatrixType & out)
{
    out = in.array().exp();
    VectorType sum_exp = out.colwise().sum();
    for(int i = 0 ; i < out.rows() ; ++i)
        out.row(i) = out.row(i).array() / sum_exp.transpose().array();
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

class neural_net{

public:
    class early_stopper : public umintl::stopping_criterion<BackendType>{
    public:
        early_stopper(neural_net const & net, MatrixType const & validation_data, LabelType const & validation_labels) : net_(net), validation_data_(validation_data), validation_labels_(validation_labels), best_cost_(INFINITY){ }
        bool operator()(umintl::detail::optimization_context<BackendType> & c){
            if(c.iter()%10==0){
                net_.set_weights(c.x());
                net_.feedforward(validation_data_);
                ScalarType cross_entropy = net_.get_cost(validation_labels_);
                bool stop = (cross_entropy/best_cost_)>1.05;
                if(cross_entropy<best_cost_){
                    best_cost_ = cross_entropy;
                    best_x_=c.x();
                }
                best_cost_ = std::min(best_cost_, cross_entropy);

                unsigned int n_misclassified = 0;
                for(int j = 0 ; j < validation_labels_.rows() ; ++j){
                    int prediction;
                    net_.A.back().col(j).maxCoeff(&prediction);
                    n_misclassified+= static_cast<unsigned int>(validation_labels_(j)!=prediction);
                }
                std::cout << "Misclassified on test set : " << (float)n_misclassified/validation_labels_.rows()*100 << "%" << std::endl;

                return stop;
            }
            return false;
        }
        VectorType const & best_x(){ return best_x_; }

    private:
        neural_net const & net_;

        MatrixType const & validation_data_;
        LabelType const & validation_labels_;

        VectorType best_x_;
        ScalarType best_cost_;
    };

    early_stopper * create_early_stopping(MatrixType const & validation_data, LabelType const & validation_labels)
    {
        return new early_stopper(*this,validation_data,validation_labels);
    }

private:


    template<class T>
    void feedforward(Eigen::MatrixBase<T> const & data) const{
        for(std::size_t L = 0 ; L < n_layers_; ++L){
            if(L==0)
                Z[L] = weights[L]*data;
            else
                Z[L] = weights[L]*A[L-1];
            Z[L].colwise() += bias[L];

            if(L==n_layers_-1)
                softmax(Z[L],A[L]);
            else
                sigmoid(Z[L],A[L]);
        }
    }

    template<class T, class U>
    void backpropagate(Eigen::MatrixBase<T> const & data, Eigen::MatrixBase<U> const & labels) const{
        for(int L = (int)n_layers_-1 ; L>=0 ; --L){
            //Compute delta
            if(L==(int)n_layers_-1){
                D[L] = A[L];
                for(int j = 0 ; j < data.cols() ; ++j){
                    D[L](labels(j),j)-=1;
                }
            }
            else{
                D[L] = weights[L+1].transpose()*D[L+1];
                D[L] = D[L].array()*A[L].array()*(1-A[L].array());
            }

            //Compute derivatives
            if(L==0)
                dweights[L] = D[L]*data.transpose() + lambda_*weights[L];
            else
                dweights[L] = D[L]*A[L-1].transpose() + lambda_*weights[L];

            dbias[L] = D[L].rowwise().sum();
        }
    }

    template<class T>
    ScalarType get_cost(Eigen::MatrixBase<T> const & labels) const{
        ScalarType cross_entropy = 0;
        for(int j = 0 ; j < A.back().cols() ; ++j){
            cross_entropy-=log(A.back()(labels(j),j));
        }
        ScalarType regularizer = 0;
        for(std::size_t L = 0 ; L < n_layers_ ; ++L)
            regularizer+=0.5*lambda_*weights[L].array().pow(2).sum();
        return cross_entropy+regularizer;
    }

public:
    neural_net(MatrixType const & data, LabelType const & labels,
       std::vector<std::size_t> const & hidden_sizes, ScalarType lambda = 0.01) :
        data_(data), labels_(labels), lambda_(lambda)
    {
        layer_sizes_.push_back(data.rows());
        for(std::size_t i = 0 ; i < hidden_sizes.size() ; ++i)
            layer_sizes_.push_back(hidden_sizes[i]);
        layer_sizes_.push_back(labels.maxCoeff()+1);
        n_layers_ = layer_sizes_.size() - 1;
        weights.resize(n_layers_);
        bias.resize(n_layers_);
        dweights.resize(n_layers_);
        dbias.resize(n_layers_);
        A.resize(n_layers_);
        Z.resize(n_layers_);
        D.resize(n_layers_);

        for(std::size_t L = 0 ; L < n_layers_; ++L){
            weights[L].resize(layer_sizes_[L+1], layer_sizes_[L]);
            bias[L].resize(layer_sizes_[L+1]);
        }
    }

    void set_weights(VectorType const & X) const{
        std::size_t offset = 0;
        for(std::size_t L = 0 ; L < n_layers_ ; ++L){
            for(int i = 0 ; i < weights[L].rows() ; ++i)
                for(int j = 0 ; j < weights[L].cols() ; ++j)
                    weights[L](i,j) = X[offset++];
            for(int i = 0 ; i < bias[L].rows() ; ++i)
                bias[L](i) = X[offset++];
        }

    }

    float misclassified_rate(MatrixType const & data, LabelType const & labels){
        feedforward(data);
        unsigned int n_misclassified = 0;
        for(int j = 0 ; j < labels.rows() ; ++j){
            int prediction;
            A.back().col(j).maxCoeff(&prediction);
            n_misclassified+= static_cast<unsigned int>(labels(j)!=prediction);
        }
        return (float)n_misclassified/labels.rows()*100;
    }

    std::size_t n_params() const{
        std::size_t res=0;
        for(std::size_t L = 0 ; L < n_layers_ ; ++L)
            res+=weights[L].rows()*weights[L].cols() + bias[L].rows();
        return res;
    }

    void operator()(VectorType const & X, ScalarType * val, VectorType * grad)const{
        set_weights(X);
        feedforward(data_);
        if(val)
            *val = get_cost(labels_);


//        unsigned int n_misclassified = 0;
//        for(int j = 0 ; j < labels_.rows() ; ++j){
//            unsigned int prediction;
//            A.back().col(j).maxCoeff(&prediction);
//            n_misclassified+= static_cast<unsigned int>(labels_(j)!=prediction);
//        }
//        std::cout << "Misclassified on training set : " << (float)n_misclassified/labels_.rows()*100 << "%" << std::endl;


        if(grad){
            backpropagate(data_,labels_);
            std::size_t offset = 0;
            for(std::size_t L = 0 ; L < n_layers_ ; ++L){
                for(int i = 0 ; i < dweights[L].rows() ; ++i)
                    for(int j = 0 ; j < dweights[L].cols() ; ++j)
                        (*grad)[offset++] = dweights[L](i,j);
                for(std::size_t i = 0 ; i < layer_sizes_[L+1] ; ++i)
                    (*grad)[offset++] = dbias[L](i);
            }
        }
    }

private:
    friend class early_stopper;

    mutable std::vector<MatrixType> weights;
    mutable std::vector<VectorType> bias;

    mutable std::vector<MatrixType> dweights;
    mutable std::vector<VectorType> dbias;

    mutable std::vector<MatrixType> Z;
    mutable std::vector<MatrixType> A;

    mutable std::vector<MatrixType> D;

    MatrixType const & data_;
    LabelType const & labels_;

    ScalarType lambda_;

    mutable std::vector<std::size_t> layer_sizes_;
    std::size_t n_layers_;
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

//    MatrixType training_data = MatrixType::Random(5,100);
//    LabelType training_label = LabelType::Zero(100);
//    for(int i = 0 ; i < 10 ; ++i)
//        training_label(i) = rand()%10;
//    MatrixType testing_data = MatrixType::Random(5,100);
//    LabelType testing_label = LabelType::Zero(100);
//    for(int i = 0 ; i < 10 ; ++i)
//        testing_label(i) = rand()%10;

    std::cout << "done!" << std::endl;



    std::cout << "#Initializing the network..." << std::flush;
    std::vector<std::size_t> hidden_sizes;
    hidden_sizes.push_back(10);
    hidden_sizes.push_back(20);
    neural_net network(training_data,training_label,hidden_sizes,1);
    VectorType Res(network.n_params());
    for(int i = 0 ; i < Res.rows() ; ++i)
        Res(i) = (ScalarType)rand()/RAND_MAX - 0.5;

    std::cout << "done!" << std::endl;

    //std::cout << "#Checking gradient..." << std::flush;
    std::cout << "Maximum relative error : " << umintl::check_grad<BackendType>(network,Res,Res.rows(),1e-6) << std::endl;

    umintl::minimizer<BackendType> optimization;
    //optimization.direction = new umintl::conjugate_gradient<BackendType>();
    //optimization.direction = new umintl::steepest_descent<BackendType>();
    //optimization.direction = new umintl::quasi_newton<BackendType>(new umintl::lbfgs<BackendType>(8));
    neural_net::early_stopper * stop = network.create_early_stopping(testing_data,testing_label);
    optimization.stopping_criterion = stop;
    optimization.max_iter = 1000;
    optimization.verbosity_level=2;
    optimization(Res,network,Res,Res.rows());
    network.set_weights(stop->best_x());
    std::cout << "Training complete!" << std::endl;
    std::cout << "Test error rate : " << network.misclassified_rate(testing_data, testing_label) << "%" << std::endl;
    std::cout << "Training error rate : " << network.misclassified_rate(training_data, training_label) << "%" << std::endl;
}
