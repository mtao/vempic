
size_t VEMMesh3::coefficient_size(size_t order) const {
    // \sum_{d=0}^order \sum_{f=0}^{order-d} \sum_{k=0}^{order-d-f} 1
    // \sum_{d=0}^order \sum_{f=0}^{order-d} order-d-f+1
    // \sum_{d=0}^order (order-d+1)^2 - \sum_{f=0}^{order-d} f
    // \sum_{d=0}^order (order-d+1)^2 - (order-d)(order-d+1)/2
    // \sum_{d=0}^order (d+1)^2 - (d)(d+1)/2
    // \sum_{d=0}^order (d+1)(d+1 - d/2)
    // \sum_{d=0}^order (d+1)(d/2+1)
    // \sum_{d=0}^order d^2/2 + 3/2 d + 1
    // order*(order+1)*(2*order+1)/12 + 3*order * (order+1) / 4 + order + 1
    // (order+1) * (2*order^2+order) / 12 + (3 * order * order+ 7 * order + 4)/ 4
    // (order+1) * (2*order^2+order) / 12 + (3 * order + 4) * (order + 1)/ 4
    // (order+1) * (2*order^2+order+9*order + 12) / 12
    // (order+1) * (2*order^2+10*order + 12) / 12
    // (order+1) * (order^2+5*order + 6) / 6
    // (order+1) * (order+2) * (order+3) / 6
    return (order + 1) * (order + 2) * (order + 3) / 6;
}
