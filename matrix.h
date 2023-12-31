#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn_utils.h"
#include <string.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    size_t rows;
    size_t cols;
    float *es;
} Mat;

typedef struct
{
    Mat inputs;
    Mat outputs;
} DataMatrix;
#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + j]
#define MAT_PRINT(m, p) mat_print((m), #m, p);
Mat mat_alloc(size_t rows, size_t cols);
void mat_assert(Mat m);
void mat_rand(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_fill(Mat m, float x);
void mat_copy(Mat dest, Mat src);
void mat_act(Mat m, AF af);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat src);
void mat_zero(Mat m);
void row_copy(Mat dest, Mat src, size_t row);
Mat csvToMat(const char *filename);
Mat mat_sub(Mat m, int rows, int cols, int col_offset);

Mat mat_alloc(size_t rows, size_t cols)
{
    assert(rows > 0 && "rows can not be zero or negative");
    assert(cols > 0 && "cols can not be zero or negative");

    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = (float *)malloc(sizeof(float) * rows * cols);
    assert(m.es != NULL && "Memory allocation failed");
    return m;
}

void mat_assert(Mat m)
{
    assert(m.rows > 0 && "mat is not initialized");
    assert(m.cols > 0 && "mat is not initialized");
    assert(m.es != NULL && "mat is not intialized");
}
void mat_rand(Mat m)
{
    mat_assert(m);
    for (size_t row = 0; row < m.rows; row++)
        for (size_t col = 0; col < m.cols; col++)
            MAT_AT(m, row, col) = rand_float();
}

void mat_print(Mat m, const char *name, size_t padding)
{
    mat_assert(m);
    printf("\n%*s%s = [\n", 1 * padding, "", name);
    for (size_t row = 0; row < m.rows; row++)
    {
        printf("%*s", 2 * padding, "");
        for (size_t col = 0; col < m.cols; col++)
        {
            printf("  %f", MAT_AT(m, row, col));
        }
        printf("\n");
    }
    printf("%*s]", 1 * padding, "");
}
void mat_fill(Mat m, float x)
{
    mat_assert(m);
    for (size_t row = 0; row < m.rows; row++)
        for (size_t col = 0; col < m.cols; col++)
            MAT_AT(m, row, col) = x;
}
void mat_copy(Mat dest, Mat src)
{
    assert(dest.cols == src.cols && "row and cols of dest and src  does not mathed");
    assert(dest.rows == src.rows && "row and cols of dest and src  does not mathed");
    for (size_t row = 0; row < dest.rows; row++)
        for (size_t col = 0; col < dest.cols; col++)
            MAT_AT(dest, row, col) = MAT_AT(src, row, col);
}
void mat_act(Mat m, AF af)
{
    mat_assert(m);
    for (size_t row = 0; row < m.rows; row++)
        for (size_t col = 0; col < m.cols; col++)
            MAT_AT(m, row, col) = act(MAT_AT(m, row, col), af);
}
void mat_dot(Mat dest, Mat a, Mat b)
{
    mat_assert(dest);
    mat_assert(a);
    mat_assert(b);
    assert(a.cols == b.rows && "first matrix cols are not equal to second matrix rows");
    assert(dest.rows == a.rows && "dest matrix rows are not equal to first matrix rows");
    assert(dest.cols == b.cols && "dest matrix cols are not equla to second matrix cols");

    for (size_t row = 0; row < dest.rows; row++)
    {
        for (size_t col = 0; col < dest.cols; col++)
        {
            MAT_AT(dest, row, col) = 0;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dest, row, col) += MAT_AT(a, row, k) * MAT_AT(b, k, col);
            }
        }
    }
}
void mat_sum(Mat dest, Mat src)
{
    mat_assert(dest);
    mat_assert(src);
    assert(dest.rows == src.rows && "dest matrix rows are not equal to src matrix rows");
    assert(dest.cols == src.cols && "dest matrix cols are not equla to src matrix cols");
    for (size_t row = 0; row < dest.rows; row++)
        for (size_t col = 0; col < dest.cols; col++)
            MAT_AT(dest, row, col) += MAT_AT(src, row, col);
}
void mat_zero(Mat m)
{
    mat_assert(m);
    for (size_t row = 0; row < m.rows; row++)
        for (size_t col = 0; col < m.cols; col++)
            MAT_AT(m, row, col) = 0.0;
}

void row_copy(Mat dest, Mat src, size_t row)
{
    assert(row >= 0 && "invalid row number");
    assert(dest.rows == 1);
    mat_assert(dest);
    mat_assert(src);
    assert(dest.cols == src.cols && "dest cols are not equal to src cols");
    assert(row < src.rows && "row number exceded");

    for (size_t j = 0; j < src.cols; j++)
        MAT_AT(dest, 0, j) = MAT_AT(src, row, j);
}

Mat mat_sub(Mat m, int rows, int cols, int col_offset)
{
    mat_assert(m);
    assert(rows > 0 && "invlaid row_max value");
    assert(rows <= m.rows && "invalid input for mat_sub rows is larger thar matrix rows");
    assert(cols > 0 && "invlaid row_max value");
    assert(cols <= m.cols && "invalid input for mat_sub row_max is larger thar matrix rows");
    assert(col_offset < m.cols && "invlaid value for col_offset");
    assert(col_offset >= 0 && "invlaid value for col_offset");
    Mat x = mat_alloc(rows, cols);
    for (int i = 0; i < x.rows; i++)
        for (int j = 0; j < x.cols; j++)
            MAT_AT(x, i, j) = MAT_AT(m, i, j + col_offset);

    return x;
}

Mat csvToMat(const char *filename)
{
    FILE *file = fopen(filename, "r");
    char buffer[1024];
    assert(file != NULL && "can not open a csv file");
    fgets(buffer, sizeof(buffer), file);
    int rows = 0;
    int cols = 0;
    char *token = strtok(buffer, ",");
    char *endptr;
    while (token != NULL)
    {
        token = strtok(NULL, ",");
        cols++;
    }
    while (fgets(buffer, sizeof(buffer), file))
    {
        rows++;
    }

    Mat m = mat_alloc(rows, cols);
    rows = 0;
    cols = 0;
    fseek(file, 0, SEEK_SET);
    double value;

    fgets(buffer, sizeof(buffer), file);
    while (fgets(buffer, sizeof(buffer), file))
    {
        token = strtok(buffer, ",");
        cols = 0;
        while (token != NULL)
        {
            MAT_AT(m, rows, cols) = (float)atof(token);
            token = strtok(NULL, ",");
            cols++;
        }
        rows++;
    }
    return m;
}

#endif