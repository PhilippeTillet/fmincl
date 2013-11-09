#ifndef UMINTL_MODEL_TYPE_FORWARDS_H
#define UMINTL_MODEL_TYPE_FORWARDS_H

#include <cstddef>

namespace umintl{

    class model_type_base{
    public:
        virtual ~model_type_base(){ }
        virtual void update(std::size_t i) = 0;
    };

}

#endif
