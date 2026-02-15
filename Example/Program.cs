using SharpGrad;
using SharpGrad.DifEngine.Loss;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using SharpGrad.NN;

internal class Program
{
    private static void Main(string[] args)
    {
        DeviceCpu cpu = new();

        Console.SetWindowSize(DataSet.N * 2 + 4, DataSet.N + 4);

        Dimension batch = new(nameof(batch), 400);
        var v = DataSet.GetDataSet(batch.Size);

        Dimension input = new(nameof(input), 2);
        Dimension hidden = new(nameof(hidden), 8);
        Dimension output = Dimension.Scalar;
        MLP<float> cerebrin = new([input, hidden, output]);

        int epochs = 30;

        DataSet.Data[] preds = new DataSet.Data[batch.Size];

        float lr = 1e-4f;
        // List of input data
        float[,] xData = new float[batch.Size, input.Size];
        for(int b = 0; b < batch.Size; b++)
        {
            for(int i = 0; i < input.Size; i++)
            {
                xData[b, i] = v[b].X[i];
            }
        }
        Variable<float> X = new("X", xData, batch, input);

        // List of ground truth data
        var ygt = v.Select(d => (float)d.Y[0]).ToArray();
        Variable<float> Ygt = new(nameof(Ygt), ygt, (output, batch));

        // Build execution expression graph (no computation done here)
        Value<float> Y = cerebrin.Forward(X);
        Value<float> loss = Loss.MSE(Y, Ygt, output);
        loss = VMath.Sum(loss, batch) / batch.Size;
        loss.IsOutput = true;

        // Training loop
        float minLoss = float.MaxValue;
        DateTime lastShow = DateTime.Now;
        for (int i = 0; i < epochs; i++)
        {
            Console.SetCursorPosition(0, 0);
            Console.WriteLine($"LR: {lr:E2} | Epoch: {i} / {epochs}");
            // Forward and backward pass
            //loss.Forward();
            cpu.Forward(loss);
            cpu.Backward(loss);

            // Build prediction data
            var yData = (IReadOnlyDataBuffer<float>)Y.Data;
            for (int b = 0; b < batch.Size; b++)
            {
                float d = yData[b];
                int val = Math.Abs(d - 1) < Math.Abs(d - 2) ? 1 : 2;
                preds[b] = new(v[b].X, [val]);
            }

            // Update weights
            cerebrin.Step(lr);
            // Reset gradients
            cpu.ResetGradient(loss);

            // Print loss and scatter plot
            float lossValue = loss.Data[0];
            Console.WriteLine($"Loss: {lossValue:E3} / {minLoss:E3}");
            if ((DateTime.Now - lastShow).TotalMilliseconds > 125)
            {
                lastShow = DateTime.Now;
                DataSet.Scatter(v, preds);
            }
            if (minLoss > lossValue)
            {
                minLoss = lossValue;
            }
        }
    }
}












// Value<float> a = new Value<float>(1.5f,"a");
// Value<float> b = new Value<float>(2.0f,"b");
// Value<float> c = new Value<float>(6.0f,"b");

// Value<float> d=(a+b*c);
// Value<float> e=d/(new Value<float>(2.0f,"2"));
// Value<float> f=e.Pow(new Value<float>(2.0f,"2"));
// Value<float> g=f.ReLU();   

// g.Grad=1.0f;
// g.Backpropagate();

// Console.WriteLine(a.Grad);
// Console.WriteLine(b.Grad);
// Console.WriteLine(c.Grad);

// Value<float> j= new Value<float>(0.5f,"j");
// Value<float> k= j.Tanh();
// Value<float> l= k.Sigmoid();
// Value<float> m= l.LeakyReLU(1.0f);
// m.Grad=1.0f;
// m.Backpropagate();
// Console.WriteLine(j.Grad);
// Console.WriteLine(m.Data);


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
