#pragma once
namespace vem {

    namespace internal {
    template <int D>
    struct VEMMeshType {
        using type = void;
    };
    }
    template <int D>
    using VEMMesh = typename internal::VEMMeshType<D>::type;

}
