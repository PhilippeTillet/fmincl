#ifndef UMINTL_MODEL_TYPE_DETERMINISTIC_HPP
#define UMINTL_MODEL_TYPE_DETERMINISTIC_HPP

#include "forwards.h"

namespace umintl{

    namespace model_type{

        class deterministic : public model_type_base{
        public:
            void update(std::size_t){ }
        };

    }

}

#endif
