using SharpGrad;
using SharpGrad.Activation;
using System.Numerics;

namespace SharpGrad.NN
{
    public class Layer<TType>
        where TType : struct, INumber<TType>
    {
        public static readonly Random Rand = new();

        public readonly Dimension[] Shape;

        public readonly Variable<TType> Weights;
        public readonly Variable<TType> Biai;
        public int NeuronsCount => Shape[0].Size;
        public int Inputs => Shape[1].Size;
        public bool ActFunc;

        public Layer(Dimension output, Dimension input, bool act_func)
        {
            Shape = [output, input];

            TType[,] weights = new TType[output.Size, input.Size];
            for(int o = 0; o < output.Size; o++)
            {
                for(int i = 0; i < input.Size; i++)
                {
                    weights[o, i] = TType.CreateSaturating(Rand.NextDouble());
                }
            }
            Weights = new Variable<TType>("W", weights, Shape);

            TType[] bias = new TType[output.Size];
            for(int o = 0; o < output.Size; o++)
            {
                bias[o] = TType.CreateSaturating(Rand.NextDouble());
            }
            Biai = new Variable<TType>("B", bias, new Shape(output));

            ActFunc = act_func;
        }

        public Value<TType> Forward(Value<TType> X)
        {
            Value<TType> mul = X * Weights;
            Value<TType> sum = VMath.Sum(mul, Shape[1]);
            Value<TType> sumB = sum + Biai;
            return ActFunc ? sumB.ReLU() : sumB;
        }

        public void Step(TType lr)
        {
            var weightsData = Weights.Data;
            var weightsGrad = Weights.Grad;

            int oSize = Shape[0].Size;
            int iSize = Shape[1].Size;
            if (oSize == 1)
            {
                for (int i = 0; i < iSize; i++)
                {
                    weightsData[i] -= lr * weightsGrad[i];
                }
            }
            else if (iSize == 1)
            {
                for (int o = 0; o < oSize; o++)
                {
                    weightsData[o] -= lr * weightsGrad[o];
                }
            }
            else
            {
                for (int o = 0; o < oSize; o++)
                {
                    for (int i = 0; i < iSize; i++)
                    {
                        weightsData[o, i] -= lr * weightsGrad[o, i];
                    }
                }
            }
        }
    }
}