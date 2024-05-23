using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    public readonly partial struct Tensor<TType> : IAdditionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>
        where TType : IFloatingPoint<TType>
    {
        public static Tensor<TType> operator +(Tensor<TType> left, Tensor<TType> right)
        {
            if (left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");

            var result = new Tensor<TType>(left.Shape);

            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TType>.IsSupported)
            {
                Span<TType> leftSpan = left.data;
                Span<TType> rightSpan = right.data;
                Span<TType> resultSpan = result.data;

                while(leftSpan.Length >= Vector<TType>.Count)
                {
                    Vector<TType> leftVector = new(leftSpan);
                    Vector<TType> rightVector = new(rightSpan);

                    (leftVector + rightVector).CopyTo(resultSpan);

                    leftSpan = leftSpan[Vector<TType>.Count..];
                    rightSpan = rightSpan[Vector<TType>.Count..];
                    resultSpan = resultSpan[Vector<TType>.Count..];
                    i += Vector<TType>.Count;
                }
            }

            for (; i < left.data.Length; i++)
                result.data[i] = left.data[i] + right.data[i];

            return result;
        }
    }
}
