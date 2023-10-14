#define NN_IMPLEMENTATION
#include "nn.hpp"

class Xor
{
public:
    Mat a0;
    Mat a1;
    Mat a2;
    Mat w1;
    Mat b1;
    Mat w2;
    Mat b2;

    void print()
    {
        a0.print("a0");
        a1.print("a1");
        a2.print("a2");
        w1.print("w1");
        b1.print("b1");
        w2.print("w2");
        b2.print("b2");
    }

    void alloc()
    {
        a0.alloc(1, 2);
        a1.alloc(1, 2);
        a2.alloc(1, 1);
        w1.alloc(2, 2);
        b1.alloc(1, 2);
        w2.alloc(2, 1);
        b2.alloc(1, 1);
    }

    Xor& forward()
    {
        a1 = a0.dot(w1);
        a1.sum(b1);
        a1.sig();
        a2 = a1.dot(w2);
        a2.sum(b2);
        a2.sig();

        return *this;
    }

    float cost(Mat ti, Mat to)
    {
        assert(ti.rows == to.rows);
        assert(to.cols == a2.rows);
        std::size_t n{ ti.rows };
        
        float cost{ 0.0f };
        for (size_t i{ 0 }; i < n; ++i)
        {
            Mat x{ ti.row(i) };
            Mat y{ to.row(i) };
            this->forward();

            std::size_t q{ to.cols };
            for (size_t j{ 0 }; j < q; ++j) 
            {
                float d{ 0 };
                d = a2.elements.at(j) - y.elements.at(j);
                cost += d*d;
            }
        }
        return cost/n;
    }

    void finite_diff(Xor *g, float eps, Mat ti, Mat to)
    {
        float c{ this->cost(ti, to) };
        // std::cout << "Initial cost: " << c << '\n';
        float saved;

        for (size_t i{ 0 }; i < w1.rows; ++i) {
            for (size_t j{ 0 }; j < w1.cols; ++j) {
                saved = w1.elements.at(i*w1.stride+j);
                // w1.print("w1 before");
                w1.elements.at(i*w1.stride+j) += eps;
                // w1.print("w1 after");

                // g->w1.print("g->w1");
                // std::cout << "Current cost: " << this->cost(ti, to) << '\n';
                g->w1.elements.at(i*g->w1.stride+j) = (this->cost(ti, to) - c) / eps;
                // g->w1.print("g->w1");

                w1.elements.at(i*w1.stride+j) = saved;
            }
        }   

        for (size_t i{ 0 }; i < b1.rows; ++i) {
            for (size_t j{ 0 }; j < b1.cols; ++j) {
                // std::cout << i << " " << j << '\n';
                saved = b1.elements.at(i*b1.stride+j);
                b1.elements.at(i*b1.stride+j) += eps;
                g->b1.elements.at(i*g->b1.stride+j) = (this->cost(ti, to) - c) / eps;
                b1.elements.at(i*b1.stride+j) = saved;
            }
        }

        for (size_t i{ 0 }; i < w2.rows; ++i) {
            for (size_t j{ 0 }; j < w2.cols; ++j) {
                // std::cout << i << " " << j << '\n';
                saved = w2.elements.at(i*w2.stride+j);
                w2.elements.at(i*w2.stride+j) += eps;
                g->w2.elements.at(i*g->w2.stride+j) = (this->cost(ti, to) - c) / eps;
                w2.elements.at(i*w2.stride+j) = saved;
            }
        }

        for (size_t i{ 0 }; i < b2.rows; ++i) {
            for (size_t j{ 0 }; j < b2.cols; ++j) {
                // std::cout << i << " " << j << '\n';
                saved = b2.elements.at(i*b2.stride+j);
                b2.elements.at(i*b2.stride+j) += eps;
                g->b2.elements.at(i*g->b2.stride+j) = (this->cost(ti, to) - c) / eps;
                b2.elements.at(i*b2.stride+j) = saved;
            }
        }
    }

    void learn(Xor g, float rate)
    {
        for (size_t i{ 0 }; i < w1.rows; ++i) {
            for (size_t j{ 0 }; j < w1.cols; ++j) {
                w1.elements.at(i*w1.stride+j) -= rate*g.w1.elements.at(i*g.w1.stride+j);
            }
        }   

        for (size_t i{ 0 }; i < b1.rows; ++i) {
            for (size_t j{ 0 }; j < b1.cols; ++j) {
                b1.elements.at(i*b1.stride+j) -= rate*g.b1.elements.at(i*g.b1.stride+j);
            }
        }

        for (size_t i{ 0 }; i < w2.rows; ++i) {
            for (size_t j{ 0 }; j < w2.cols; ++j) {
                w2.elements.at(i*w2.stride+j) -= rate*g.w2.elements.at(i*g.w2.stride+j);
            }
        }

        for (size_t i{ 0 }; i < b2.rows; ++i) {
            for (size_t j{ 0 }; j < b2.cols; ++j) {
                b2.elements.at(i*b2.stride+j) -= rate*g.b2.elements.at(i*g.b2.stride+j);
            }
        }
    }
};

std::vector<float> td {
    0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f,
};

int main()
{

    size_t stride{ 3 };
    size_t n{ std::size(td)/stride };
    Xor m;
    m.alloc();
    Xor g;
    g.alloc();

    m.w1.rand(0, 1);
    m.b1.rand(0, 1);
    m.w2.rand(0, 1);
    m.b2.rand(0, 1);

    m.print();
    g.print();

    return 0;

    Mat ti;
    ti.alloc(n, 2, stride, td);
    // ti.print("ti");

    Mat to;
    to.alloc(n, 1, stride, (std::vector<float>(td.begin() + 2, td.end())));
    // to.print("to");

    float eps = 1e-1;
    float rate = 1e-1;

    std::cout << "Cost = " << m.cost(ti, to) << '\n';

    for (size_t i{ 0 }; i < 100*1000; ++i)
    {
        m.finite_diff(&g, eps, ti, to);
        m.learn(g, rate);
        std::cout << i << ": Cost = " << m.cost(ti, to) << '\n';
    }
    // g.print();
    // m.print();
    // std::cout << "Cost = " << m.cost(ti, to) << '\n';


    // for (std::size_t i{ 0 }; i < 2; ++i) {
    //     for (std::size_t j{ 0 }; j < 2; ++j) {
    //         m.a0.elements.at(i*m.a0.cols+0) = i;
    //         m.a0.elements.at(i*m.a0.cols+1) = j;
    //         m.forward();
    //         float y{ m.a2.elements.at(0) };

    //         std::cout << i << " ^ " << j << " = " << y << '\n';
    //     }
    // }

    return 0;
}