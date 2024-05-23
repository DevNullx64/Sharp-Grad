using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    public readonly partial struct Tensor<TType>
        where TType : IFloatingPoint<TType>
    {
        private readonly TType[] data;
        private readonly TType[]? gradients;
        public readonly Shape Shape;

        public readonly TType this[int indice, params int[] indices]
        {
            get => data[Shape.GetFlattenedIndex(indice, indices)];
            set => data[Shape.GetFlattenedIndex(indice, indices)] = value;
        }

        public Tensor(Shape shape)
        {
            data = new TType[shape.Aggregate(1, (a, b) => a * b)];
            gradients = new TType[data.Length];
            Shape = shape;
        }

        public readonly void AddGradient(Tensor<TType> gradient)
        {
            if(gradient.Shape != Shape)
                throw new ArgumentException($"Expected gradient shape {Shape}, got {gradient.Shape}");

            if (gradients != null)
            {
                for (int i = 0; i < gradients.Length; i++)
                    gradients[i] += gradient.data[i];
            }
        }
    }
}
