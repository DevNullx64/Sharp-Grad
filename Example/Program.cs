
using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.NN;
using SharpGrad.Tensors;

var v = DataSet.GetDataSet(400);
Console.WriteLine("Dataset:");
DataSet.Scatter(v);

// ! \\ PoC // ! \\

Random rnd = new();
Tensor<float> ta = new(256, 256, 256);
for(int d = 0; d < ta.Shape[0]; d++)
    for (int i = 0; i < ta.Shape[1]; i++)
        for (int j = 0; j < ta.Shape[2]; j++)
            ta[d, i, j] = (float)rnd.NextDouble();

Tensor<float> tb = new(256, 256, 256);
for (int d = 0; d < tb.Shape[0]; d++)
    for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
            tb[d, i, j] = (float)rnd.NextDouble();

Tensor<float> tc = new(256, 256, 256);

Tensor<float> ty = new(256, 256, 256);

// Test dynamic operations
for (int d = 0; d < tb.Shape[0]; d++)
    for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
        {
            ty[d, i, j] += ta[d, i, j] + tb[d, i, j];
            ty[d, i, j] += ta[d, i, j] * tb[d, i, j];
            ty[d, i, j] += ta[d, i, j] / tb[d, i, j];
            ty[d, i, j] += ta[d, i, j] - tb[d, i, j];
        }

tc = Tensor<float>.ExecGpu([Operation.Add, Operation.Mul, Operation.Div, Operation.Sub], ta, tb);

float diff = 0;
float min = float.MaxValue;
float max = float.MinValue;

int size = ta.Shape.Size;
for (int d = 0; d < tc.Shape[0]; d++)
    for (int i = 0; i < tc.Shape[1]; i++)
        for (int j = 0; j < tc.Shape[2]; j++)
        {
            float diff_ = Math.Abs(tc[d, i, j] - ty[d, i, j]);
            diff += diff_ / size;
            if (diff_ < min)
                min = diff_;
            if (diff_ > max)
                max = diff_;
        }
Console.WriteLine($"dynamic test passed: error mean={diff}, min={min}, max={max}");

// Test Addition
for (int d = 0; d < tb.Shape[0]; d++)
   for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
             ty[d, i, j] = ta[d, i, j] + tb[d, i, j];
tc = ta + tb;

for (int d = 0; d < tc.Shape[0]; d++)
    for (int i = 0; i < tc.Shape[1]; i++)
        for (int j = 0; j < tc.Shape[2]; j++)
            if(tc[d, i, j] != ty[d, i, j])
                Console.WriteLine($" !!!!! Error: [{d},{i},{j}]{tc[d, i, j]} != [{d},{i},{j}]{ty[d, i, j]} (expected)");
Console.WriteLine("Addition test passed.");

// Test Subtraction
for (int d = 0; d < tb.Shape[0]; d++)
    for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
            ty[d, i, j] = ta[d, i, j] - tb[d, i, j];
tc = ta - tb;

for (int d = 0; d < tc.Shape[0]; d++)
    for (int i = 0; i < tc.Shape[1]; i++)
        for (int j = 0; j < tc.Shape[2]; j++)
            if (tc[d, i, j] != ty[d, i, j])
                Console.WriteLine($" !!!!! Error: [{d},{i},{j}]{tc[d, i, j]} != [{d},{i},{j}]{ty[d, i, j]} (expected)");
Console.WriteLine("Subtraction test passed.");

// Test Multiplication
for (int d = 0; d < tb.Shape[0]; d++)
    for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
            ty[d, i, j] = ta[d, i, j] * tb[d, i, j];
tc = ta * tb;

for (int d = 0; d < tc.Shape[0]; d++)
    for (int i = 0; i < tc.Shape[1]; i++)
        for (int j = 0; j < tc.Shape[2]; j++)
            if (tc[d, i, j] != ty[d, i, j])
                Console.WriteLine($" !!!!! Error: [{d},{i},{j}]{tc[d, i, j]} != [{d},{i},{j}]{ty[d, i, j]} (expected)");
Console.WriteLine("Multiplication test passed.");

// Test Division
for (int d = 0; d < tb.Shape[0]; d++)
    for (int i = 0; i < tb.Shape[1]; i++)
        for (int j = 0; j < tb.Shape[2]; j++)
            ty[d, i, j] = ta[d, i, j] / tb[d, i, j];
tc = ta / tb;

for (int d = 0; d < tc.Shape[0]; d++)
    for (int i = 0; i < tc.Shape[1]; i++)
        for (int j = 0; j < tc.Shape[2]; j++)
            if (tc[d, i, j] != ty[d, i, j])
                Console.WriteLine($" !!!!! Error: [{d},{i},{j}]{tc[d, i, j]} != [{d},{i},{j}]{ty[d, i, j]} (expected)");
Console.WriteLine("Division test passed.");

Console.WriteLine($"//// Finish \\\\\\\\");
// ! \\ PoC // ! \\
return;

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












//Value a = new Value(1.5,"a");
//Value b = new Value(2.0,"b");
//Value c = new Value(6.0,"b");

// value d=(a+b*c);
// value e=d/(new value(2.0,"2"));
// value f=e^(new value(2.0,"2"));
// value g=f.relu();   

// g.grad=1.0;
// g.backpropagate();

// Console.WriteLine(a.grad);
// Console.WriteLine(b.grad);
// Console.WriteLine(c.grad);

// value j= new value(0.5,"j");
// value k= j.tanh();

// k.grad=1.0;
// k.backpropagate();
// Console.WriteLine(j.grad);
// Console.WriteLine(k.data);


// MLP cerebrin = new MLP(4,new List<int>{4,16,16,1});

// List<value> X = new List<value>();
// X.Add(new value(1.0,"x1"));
// X.Add(new value(2.0,"x2"));
// X.Add(new value(3.0,"x3"));
// X.Add(new value(4.0,"x4"));

// List<value> Y = cerebrin.forward(X);
// Console.WriteLine(Y[0].data);