#ifndef FMINCL_TYPELIST_HPP
#define FMINCL_TYPELIST_HPP

/** @file typelist.hpp
 *  @brief Generic implementation of a typelist. Experimental.
 *
 */

namespace fmincl
{

    template <class T,class U>
    struct typelist
    {
      typedef T Head;
      typedef U Tail;
    };

    struct NullType{ };

    template
    <
        typename T1  = NullType, typename T2  = NullType, typename T3  = NullType,
        typename T4  = NullType, typename T5  = NullType, typename T6  = NullType,
        typename T7  = NullType, typename T8  = NullType, typename T9  = NullType,
        typename T10 = NullType, typename T11 = NullType, typename T12 = NullType,
        typename T13 = NullType, typename T14 = NullType, typename T15 = NullType,
        typename T16 = NullType, typename T17 = NullType, typename T18 = NullType
    >
    struct make_typelist
    {
      private:
        typedef typename make_typelist
        <
            T2 , T3 , T4 ,
            T5 , T6 , T7 ,
            T8 , T9 , T10,
            T11, T12, T13,
            T14, T15, T16,
            T17, T18
        >
        ::type TailResult;

      public:
        typedef typelist<T1, TailResult> type;
    };

    template <>
    struct make_typelist<>
    {
        typedef NullType type;
    };


}

#endif

