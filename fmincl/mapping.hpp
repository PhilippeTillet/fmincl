/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_MAPPING_HPP_
#define FMINCL_MAPPING_HPP_

#include <typeinfo>

namespace fmincl{

template<class BackendType, class BaseType, class T1, class T2=void, class T3=void>
struct implementation_of{
    template<class ContextType>
    static typename BaseType::template implementation<BackendType> * create(BaseType const & tag, ContextType & context){
        if(typeid(tag)==typeid(T1))
            return new typename T1::template implementation<BackendType>(static_cast<T1 const &>(tag),context);
        else
            return implementation_of<BackendType,BaseType,T2,T3>::create(tag,context);
    }
};

template<class BackendType, class BaseType>
struct implementation_of<BackendType,BaseType,void>{
    template<class ContextType>
    static typename BaseType::template implementation<BackendType> * create(BaseType const &, ContextType &){ return NULL; }
};

}

#endif
