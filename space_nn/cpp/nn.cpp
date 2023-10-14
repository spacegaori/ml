#define NN_IMPLEMENTATION
#include "nn.hpp"

// {2, 2, 1}

using Neuron = std::vector<Matrix>;
using Index = Neuron::size_type;

class NN
{
public:
    void alloc(Index* arch, Index arch_count)
    {
        nn.m_count  = arch_count - 1;
        nn.m_ws = Neuron;
    }

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

    void finite_diff(Xor *g, float eps, Mat ti, Mat to)
    {
        float c{ cost(ti, to) };
        float saved;
        Index w1_rows{ w1.getRows() };
        Index w1_cols{ w1.getCols() };
        for (Index i{ 0 }; i < w1_rows; ++i) {
            for (Index j{ 0 }; j < w1_cols; ++j) {
                saved = w1.at(i*w1_cols+j);
                w1.sum_at(i*w1_cols+j, eps);
                g->w1.set(i*g->w1.getCols()+j, ((cost(ti, to) - c) / eps));
                w1.set(i*w1_cols+j, saved);
            }
        }
        Index b1_rows{ b1.getRows() };
        Index b1_cols{ b1.getCols() };
        for (Index i{ 0 }; i < b1_rows; ++i) {
            for (Index j{ 0 }; j < b1_cols; ++j) {
                saved = b1.at(i*b1_cols+j);
                b1.sum_at(i*b1_cols+j, eps);
                g->b1.set(i*g->b1.getCols()+j, ((cost(ti, to) - c) / eps));
                b1.set(i*b1_cols+j, saved);
            }
        }
        Index w2_rows{ w2.getRows() };
        Index w2_cols{ w2.getCols() };
        for (Index i{ 0 }; i < w2_rows; ++i) {
            for (Index j{ 0 }; j < w2_cols; ++j) {
                saved = w2.at(i*w2_cols+j);
                w2.sum_at(i*w2_cols+j, eps);
                g->w2.set(i*g->w2.getCols()+j, ((cost(ti, to) - c) / eps));
                w2.set(i*w2_cols+j, saved);
            }
        }

        Index b2_rows{ b2.getRows() };
        Index b2_cols{ b2.getCols() };
        for (Index i{ 0 }; i < b2_rows; ++i) {
            for (Index j{ 0 }; j < b2_cols; ++j) {
                saved = b2.at(i*b2_cols+j);
                b2.sum_at(i*b2_cols+j, eps);
                g->b2.set(i*g->b2.getCols()+j, ((cost(ti, to) - c) / eps));
                b2.set(i*b2_cols+j, saved);
            }
        }
    }

    void forward()
    {
        a1 = a0.dot(w1);
        a1.sum(b1);
        a1.sig();
        a2 = a1.dot(w2);
        a2.sum(b2);
        a2.sig();
    }

    float cost(Mat ti, Mat to)
    {
        Index n{ ti.getRows() };
        
        float cost{ 0.0f };
        for (Index i{ 0 }; i < n; ++i)
        {
            Mat x{ ti.row(i) };
            Mat y{ to.row(i) };
            forward();

            Index q{ to.getCols() };
            for (Index j{ 0 }; j < q; ++j) 
            {
                float d{ 0 };
                d = a2.at(j) - y.at(j);
                cost += d*d;
            }
        }
        return cost/n;
    }
    
    void learn(Xor g, float rate)
    {
        for (Index i{ 0 }; i < w1.getRows(); ++i) {
            for (Index j{ 0 }; j < w1.getCols(); ++j) {
                w1.sub_at(i*w1.getCols()+j, rate*g.w1.at(i*g.w1.getCols()+j));
            }
        }
        for (Index i{ 0 }; i < b1.getRows(); ++i) {
            for (Index j{ 0 }; j < b1.getCols(); ++j) {
                b1.sub_at(i*b1.getCols()+j, rate*g.b1.at(i*g.b1.getCols()+j));
            }
        }
        for (Index i{ 0 }; i < w2.getRows(); ++i) {
            for (Index j{ 0 }; j < w2.getCols(); ++j) {
                w2.sub_at(i*w2.getCols()+j, rate*g.w2.at(i*g.w2.getCols()+j));
            }
        }
        for (Index i{ 0 }; i < b2.getRows(); ++i) {
            for (Index j{ 0 }; j < b2.getCols(); ++j) {
                b2.sub_at(i*b2.getCols()+j, rate*g.b2.at(i*g.b2.getCols()+j));
            }
        }
    }
private:
    Index m_count;
    std::vector<Mat> m_ws{};
    std::vector<Mat> m_ws{};
    std::vector<Mat> m_ws{};
};


int main()
{
    Xor m, g;

    m.w1.rand(0, 1);
    m.b1.rand(0, 1);
    m.w2.rand(0, 1);
    m.b2.rand(0, 1);

    Mat td{
        4,
        3,
        Matrix{
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 1.0f,
            1.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f,
        }
    };
    td.print("td");

    Mat ti{ td.cols(0, 1) };
    ti.print("ti");
    Mat to{ td.col(2) };
    to.print("to");

    float eps{ 1e-1 };
    float rate{ 1e-1 };

    std::cout << "Cost = " << m.cost(ti, to) << '\n';
    m.finite_diff(&g, eps, ti, to);
    m.learn(g, rate);
    std::cout << "Cost = " << m.cost(ti, to) << '\n';


    // for (Index i{ 0 }; i < 100*1000; ++i)
    // {
    //     m.finite_diff(&g, eps, ti, to);
    //     m.learn(g, rate);
    //     std::cout << i << ": Cost = " << m.cost(ti, to) << '\n';
    // }

    return 0;
}
