
#include "matrix.h"

int main()
{

    Mat m1 = mat_alloc(1, 2);
    mat_fill(m1, 1);
    Mat m2 = mat_alloc(2, 1);
    mat_fill(m2, 1);
    Mat m3 = mat_alloc(1, 1);
    mat_dot(m3, m1, m2);
    MAT_PRINT(m3);
}