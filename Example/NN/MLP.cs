using SharpGrad;
using SharpGrad.DifEngine;
using System.Numerics;

namespace SharpGrad.NN
{
    public class MLP<TType>
        where TType : unmanaged, IBinaryFloatingPointIeee754<TType>
    {
        public List<Layer<TType>> Layers;
        public List<Layer<TType>> LayersDiff;
        public int Inputs;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="count"></param>
        /// <param name="count"></param>
        public MLP(params int[] count)
        {
            if (count.Length < 2)
                throw new ArgumentException($"{nameof(count)} must have at least 2 dataElements. Got {count.Length}.");

            Layers = new List<Layer<TType>>(count.Length - 1);
            LayersDiff = new List<Layer<TType>>(count.Length - 1);
            Inputs = count[0];
            Layers.Add(new Layer<TType>(count[1], Inputs, false));
            LayersDiff.Add(new Layer<TType>(count[1], Inputs, true));
            for (int i = 2; i < count.Length; i++)
            {
                Layers.Add(new Layer<TType>(count[i], count[i - 1], true));
                LayersDiff.Add(new Layer<TType>(count[i], count[i - 1], true));
            }
        }

        public List<Value<TType>> Forward(List<Value<TType>> X)
        {
            List<Value<TType>> Y;
            for(int i = 0; i < Layers.Count; i++)
            {
                Y = Layers[i].Forward(X);
                var diff = LayersDiff[i].Forward(X);
                for(int j = 0; j < Y.Count; j++)
                    Y[j].Data = Y[j].Data - diff[j].Data;

                X = Y;
            }
            return X;
        }

        public void Step(TType lr)
        {
            for (int l = 0; l < Layers.Count; l++)
            {
                foreach (var n in Layers[l].Neurons)
                {
                    foreach (var w in n.Weights)
                        w.Data -= lr * w.Grad;
                    n.Biai.Data -= lr * n.Biai.Grad;
                }
                foreach (var n in LayersDiff[l].Neurons)
                {
                    foreach (var w in n.Weights)
                        w.Data -= lr * w.Grad;
                    n.Biai.Data -= lr * n.Biai.Grad;
                }
            }
        }
    }
}