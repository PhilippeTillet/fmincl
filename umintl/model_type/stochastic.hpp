#ifndef UMINTL_MODEL_TYPE_STOCHASTIC_HPP
#define UMINTL_MODEL_TYPE_STOCHASTIC_HPP

#include "forwards.h"

namespace umintl{

    namespace model_type{

        class stochastic : public model_type_base{
        public:
            stochastic(std::size_t sample_size, std::size_t dataset_size) : sample_size_(sample_size), dataset_size_(dataset_size), sample_offset_(0){ }
            std::size_t sample_size() const{ return std::min(sample_size_, dataset_size_-sample_offset_); }
            std::size_t sample_offset() const{ return sample_offset_; }
            virtual void update(std::size_t i){ sample_offset_= ((i*sample_size_) % dataset_size_); }
        private:
            std::size_t sample_size_;
            std::size_t dataset_size_;
            std::size_t sample_offset_;
        };

    }

}

#endif
