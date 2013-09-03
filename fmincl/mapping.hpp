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

#include "tools/typelist.hpp"

namespace fmincl{

template<class _Tag, class _Impl>
struct impl_tag_mapping{
    typedef _Impl Impl;
    typedef _Tag Tag;
};

#define FMINCL_CREATE_MAPPING(name) impl_tag_mapping<name##_tag,name##_implementation<BackendType> >

template<class Types, class BaseTagType, class BaseImplementationType>
struct implementation_from_tag{
private:
    typedef typename Types::Head Head;
    typedef typename Types::Tail Tail;
public:
    static BaseImplementationType * create(BaseTagType const & tag){
        if(typeid(tag)==typeid(typename Head::Tag))
            return new typename Head::Impl(static_cast<typename Head::Tag const &>(tag));
        else
            return implementation_from_tag<Tail, BaseTagType, BaseImplementationType>::create(tag);
    }
};

template<class BaseTagType,class BaseImplementationType>
struct implementation_from_tag<NullType, BaseTagType, BaseImplementationType>{
    static BaseImplementationType * create(BaseTagType const & tag){ return NULL; }
};

}

#endif
