#ifndef NN_H
#define NN_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

float rand_float();
float sigmoidf(float x);

class Mat {
public:
    std::size_t rows {};
    std::size_t cols {};
    std::size_t stride {};
    std::vector<float> elements {};

    auto size() { return elements.size(); }
    auto capacity() { return elements.capacity(); }

    void alloc(std::size_t r, std::size_t c)
    {
        rows = r;
        cols = c;
        stride = c;
        elements.reserve(r*c);
        elements = std::vector<float>(r*c, 0.0f);
        assert(elements.size() != 0);
    }

    void alloc(std::size_t r, std::size_t c, std::vector<float> e)
    {
        rows = r;
        cols = c;
        stride = c;
        elements.reserve(r*c);
        elements = e;
        assert(elements.size() != 0);
    }

    void alloc(std::size_t r, std::size_t c, std::size_t s, std::vector<float> e)
    {
        rows = r;
        cols = c;
        stride = s;
        elements.reserve(r*c);
        elements = e;
        assert(elements.size() != 0);
    }

    void copy(Mat m)
    {
        rows = m.rows;
        cols = m.cols;
        stride = m.cols;
        elements.clear();
        elements.reserve(m.rows*m.cols);
        elements = m.elements;
    }

    void fill(float f)
    {
        for (std::size_t i{ 0 }; i < rows; ++i) {
            for (std::size_t j{ 0 }; j < cols; ++j) {
                elements.at(i*stride+j) = f;
            }
        }
    }

    Mat dot(Mat m)
    {
        Mat dst;
        dst.alloc(rows, m.cols);
        assert(cols == m.rows);
        std::size_t size{ cols };
        assert(dst.rows == rows);
        assert(dst.cols == m.cols);

        for (std::size_t i{ 0 }; i < dst.rows; ++i) {
            for (std::size_t j{ 0 }; j < dst.cols; ++j) {
                dst.elements.at(i*dst.stride+j) = 0.0f;
                for (std::size_t k{ 0 }; k < size; ++k) {
                    dst.elements.at(i*dst.stride+j) += elements.at(i*stride+k) * m.elements.at(k*m.stride+j);
                }
            }
        }

        return dst;
    }

    void rand(float low, float high)
    {
        for (std::size_t i{ 0 }; i < rows; ++i) {
            for (std::size_t j{ 0 }; j < cols; ++j) {
                elements.at(i*stride+j) = rand_float()*(high-low)+low;
            }
        }
    }

    Mat row(size_t row)
    {
        Mat m;
        m.alloc(
            1,
            cols,
            std::vector<float>(
                elements.begin()+static_cast<long>(row*cols),
                elements.end())
        );

        return m;
    }

    Mat& sum(Mat m)
    {
        assert(rows == m.rows);
        assert(cols == m.cols);

        for (std::size_t i{ 0 }; i < rows; ++i) {
            for (std::size_t j{ 0 }; j < cols; ++j) {
                elements.at(i*stride+j) += m.elements.at(i*m.stride+j);
            }
        }

        return *this;
    }

    void sig()
    {
        for (std::size_t i{ 0 }; i < rows; ++i) {
            for (std::size_t j{ 0 }; j < cols; ++j) {
                elements.at(i*stride+j) = sigmoidf(elements.at(i*stride+j));
            }
        }
    }

    void print(std::string name)
    {
        std::cout << name << " = [\n";
        for (std::size_t i{ 0 }; i < rows; ++i) {
            for (std::size_t j{ 0 }; j < cols; ++j) {
                std::cout << "    " << elements.at(i*stride+j) << ' ';
            }
            std::cout << '\n';
        }
        std::cout << "]\n";
    }
};

float rand_float()
{
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution rand(0.0, 1.0);

    return rand(mt);
}

float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::expf(-x));
}

#endif