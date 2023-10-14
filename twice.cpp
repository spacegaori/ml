#include <iostream>
#include <random>

float train[][2]
{
    { 0.0, 0.0 },
    { 1.0, 2.0 },
    { 2.0, 4.0 },
    { 3.0, 6.0 },
    { 4.0, 8.0 },
};

constexpr int train_count{ std::size(train) };

float rand_float()
{
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution rand(0.0, 1.0);

    return rand(mt);
}

float cost(float w, float b)
{
    float result{ 0.0f };
    for (auto data : train)
    {
        float x{ data[0] };
        float y{ x*w + b };
        float d{ y - data[1] };
        result += d*d;
    }
    result /= train_count;

    return result;
}

int main()
{
    float w{ rand_float()*10.0f };
    float b{ 0 };
    // float b{ rand_float()*5.0f };

    float eps{ 1e-3 };
    float rate{ 1e-3 };

    std::cout << cost(w, b) << '\n';

    for (int i{ 0 }; i < 500; ++i)
    {
        float c{ cost(w, b) };
        float dw{ (cost(w + eps, b) - c) / eps };
        // float db{ (cost(w, b + eps) - c) / eps };
        w -= rate*dw;
        // b -= rate*db;
        std::cout << "cost = " << c << ", w = " << w << ", b = " << b << '\n';
    }

    std::cout << "----------------------------------------\n";
    std::cout << "w = " << w << ", b = " << b << '\n';

    return 0;
}