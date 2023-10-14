#include <iostream>
#include <random>
#include <cmath>

struct Xor
{
    float or_w1;
    float or_w2;
    float or_b;
    float nand_w1;
    float nand_w2;
    float nand_b;
    float and_w1;
    float and_w2;
    float and_b;
};

float sigmoidf(float x)
{
    return (1.0f / (1.0f + std::expf(-x)));
}

float forward(Xor m, float x1, float x2)
{
    float a{ sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b) };   
    float b{ sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b) };   
    return sigmoidf(a*m.and_w1*x1 + b*m.and_w2*x2 + m.and_b);   
}

using Sample = float[3];

Sample or_train[]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 1.0f },
};

Sample and_train[]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f },
    { 1.0f, 1.0f, 1.0f },
};

Sample nand_train[]
{
    { 0.0f, 0.0f, 1.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 0.0f },
};

Sample* train{ nand_train };

constexpr int train_count{ std::size(or_train) };

float rand_float()
{
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution rand(0.0, 1.0);

    return rand(mt);
}

float cost(float w1, float w2, float b = 0.0f)
{
    float result{ 0.0f };
    for (int i{ 0 }; i < train_count; ++i)
    {   
        float x1{ train[i][0] };
        float x2{ train[i][1] };
        float y{ sigmoidf(x1*w1 + x2*w2 + b) };
        float d{ y - train[i][2] };
        result += d*d;
    }
    result /= train_count;

    return result;
}

int main()
{
    float w1{ rand_float() };
    float w2{ rand_float() };
    float b{ rand_float() };

    float eps{ 1e-3 };
    float rate{ 1e-1 };
    
    for(int i{ 0 }; i < 100*1000; ++i)
    {
        float c{ cost(w1, w2, b) };
        float dw1{ (cost(w1+eps, w2, b) - c) / eps };
        float dw2{ (cost(w1, w2+eps, b) - c) / eps };
        float db{ (cost(w1, w2, b+eps) - c) / eps };
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
    }

    for (int i{ 0 }; i < 2; ++i)
        for (int j{ 0 }; j < 2; ++j)
            std::cout << i << " | " << j << " = " << sigmoidf(i*w1 + j*w2 + b) << '\n';

    return 0;
}