using SharpGrad.DifEngine;
using SharpGrad.NN;

internal class Program
{
    


MLP<float> cerebrin = new(2, 8, 1);

List<Value<float>> X =
[
    v[0].X[0],
    v[0].X[1]
];
List<Value<float>> Y = cerebrin.Forward(X);


Console.WriteLine(Y[0].Data);



int epochs = 1000;
float lr = 1e-9f;

        float lastLoss = float.MaxValue;

        for (int i = 0; i < epochs; i++)
        {
            Console.SetCursorPosition(0, 0);
            Console.WriteLine("Epoch: " + i);
            Value<float> loss = Value<float>.Zero;
            List<DataSet.Data> preds = [];

    for (int j = 0; j < v.Count; j++)
    {
        X =
        [
            v[j].X[0],
            v[j].X[1]
        ];
        Y = cerebrin.Forward(X);
        List<Value<float>> Ygt =
        [
            v[j].Y[0]
        ];
        var nl = loss + Loss.MSE(Y, Ygt);
        loss = nl;

                int val;
                if (Math.Abs(Y[0].Data - 1) < Math.Abs(Y[0].Data - 2))
                {
                    val = 1;
                }
                else
                {
                    val = 2;
                }
                DataSet.Data nd = new(v[j].X, [val]);
                preds.Add(nd);
            }

            loss.Backpropagate();
            cerebrin.Step(lr);

    Console.WriteLine("Loss: " + loss.Data);
    DataSet.Scatter(preds);
    if (lastLoss > loss.Data)
    {
        lastLoss = loss.Data;
    }
    else
    {
        Console.WriteLine("Final loss: " + loss.Data);
        Console.WriteLine("Last epoch: " + i);
        Console.WriteLine("Loss is increasing. Stopping training...");
        break;
    }
}








// Result<float> a = new Result<float>(1.5f,"a");
// Result<float> b = new Result<float>(2.0f,"b");
// Result<float> c = new Result<float>(6.0f,"b");

// Result<float> d=(a+b*c);
// Result<float> e=d/(new Result<float>(2.0f,"2"));
// Result<float> f=e.Pow(new Result<float>(2.0f,"2"));
// Result<float> g=f.ReLU();   

// g.Grad=1.0f;
// g.Backpropagate();

// Console.WriteLine(a.Grad);
// Console.WriteLine(b.Grad);
// Console.WriteLine(c.Grad);

// Result<float> j= new Result<float>(0.5f,"j");
// Result<float> k= j.Tanh();
// Result<float> l= k.Sigmoid();
// Result<float> m= l.LeakyReLU(1.0f);
// m.Grad=1.0f;
// m.Backpropagate();
// Console.WriteLine(j.Grad);
// Console.WriteLine(m.DataInfos);


/***
Tested with torch:

import torch

a = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(6.0, requires_grad=True)

d = a + b * c
e = d / 2.0
f = e ** 2
g = torch.relu(f)

g.backward()

print("Gradiente de a:", a.grad)
print("Gradiente de b:", b.grad)
print("Gradiente de c:", c.grad)



def custom_leaky_relu(x, negative_slope=1.0):
    return torch.where(x > 0, x, negative_slope * x)


j = torch.tensor(0.5, requires_grad=True)
k = torch.tanh(j)
l= torch.sigmoid(k)
m = custom_leaky_relu(l, negative_slope=1.0)

m.backward()

print(j.grad)
print( m.item())
***/
