using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class SumOp<T> : IOperationN1_1<T>
        where T : unmanaged, INumber<T>, IAdditionOperators<T, T, T>
    {
        public static Shape ResultingShape(Shape operand) => operand.Count > 1 
            ? new (operand[..^1])
            : throw new ArgumentException("At least two dimensions are required.");

        public static T Exec(T[] operand)
        {
            T result = operand[0];
            for (int i = 1; i < operand.Length; i++)
                result += operand[i];
            return result;
        }

        public static void Exec(Index1D idx, ArrayView2D<T, Stride2D.DenseY> operand, int width, ArrayView1D<T, Stride1D.Dense> result)
        {   
            T sum = operand[idx, 0];
            for (int i = 1; i < width; i++)
                sum += operand[idx, i];
            result[idx] = sum;
        }

    }
}
