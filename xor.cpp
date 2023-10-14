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

Sample xor_train[]
{
    { 0.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },
    { 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 0.0f },
};

Sample* train{ xor_train };
constexpr int train_count{ 4 };

float rand_float()
{
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution rand(0.0, 1.0);

    return rand(mt);
}

float cost(Xor m)
{
    float result{ 0.0f };
    for (int i{ 0 }; i < train_count; ++i)
    {   
        float x1{ train[i][0] };
        float x2{ train[i][1] };
        float y{ forward(m, x1, x2) };
        float d{ y - train[i][2] };
        result += d*d;
    }
    result /= train_count;

    return result;
}

Xor rand_xor()
{
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    return m;
}

void print_xor(Xor m)
{
    std::cout << "or_w1 = " << m.or_w1 << '\n';
    std::cout << "or_w2 = " << m.or_w2 << '\n';
    std::cout << "or_b = " << m.or_b << '\n';

    std::cout << "nand_w1 = " << m.nand_w1 << '\n';
    std::cout << "nand_w2 = " << m.nand_w2 << '\n';
    std::cout << "nand_b = " << m.nand_b << '\n';

    std::cout << "and_w1 = " << m.and_w1 << '\n';
    std::cout << "and_w2 = " << m.and_w2 << '\n';
    std::cout << "and_b = " << m.and_b << '\n';
}

Xor learn(Xor m, Xor g, float rate) 
{
    m.or_w1 -= rate*g.or_w1;
    m.or_w2 -= rate*g.or_w2;
    m.or_b -= rate*g.or_b;

    m.nand_w1 -= rate*g.nand_w1;
    m.nand_w2 -= rate*g.nand_w2;
    m.nand_b -= rate*g.nand_b;

    m.and_w1 -= rate*g.and_w1;
    m.and_w2 -= rate*g.and_w2;
    m.and_b -= rate*g.and_b;

    return m;
}

Xor finite_diff(Xor m, float eps)
{
    Xor g;
    float c{ cost(m) };

    float saved;
    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return g;
}

int main()
{
    Xor m = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    std::cout << "cost: " << cost(m) << '\n';
    for(int i{ 0 }; i < 500*1000; ++i)
    {
        Xor g{ finite_diff(m, eps) };
        m = learn(m, g, rate);
    }
    std::cout << "cost: " << cost(m) << '\n';

    std::cout << "--------------------------------\n";
    for (int i{ 0 }; i < 2; ++i)
        for (int j{ 0 }; j < 2; ++j)
            std::cout << i << " ^ " << j << " = " << forward(m, i, j) << '\n';

    return 0;
}