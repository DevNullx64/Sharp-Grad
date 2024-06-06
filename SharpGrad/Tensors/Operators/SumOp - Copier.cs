using ILGPU;
using ILGPU.Runtime;
using System;
using System.Diagnostics;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class SuMulOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>, IAdditionOperators<T, T, T>
    {
        public static Shape ResultingShape(Shape operand1, Shape operand2)
        {
            if (operand1[1] != operand2[0])
                throw new InvalidOperationException($"Invalid shape {operand1} and {operand2}");
            return new Shape(operand1[0]);
        }

        public static T Exec(T[] operand1, T[] operand2)
        {

        }

        public static void Exec(Index1D idx, ArrayView2D<T, Stride2D.DenseY> operand, int width, ArrayView1D<T, Stride1D.Dense> result)
        {   
            T sum = operand[idx, 0];
            for (int i = 1; i < width; i++)
                sum += operand[idx, i];
            result[idx] = sum;
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView2D<T, Stride2D.DenseY> operand1, ArrayView2D<T, Stride2D.DenseY> grad1, long width)
        {
            for (int i = 0; i < width; i++)
                grad1[idx, i] += grad[idx];
        }

        public static void Backward(TensorGrad<T> @this, TensorGrad<T> operand)
        {
            Debug.Assert(@this.Shape.Count == operand.Shape.Count - 1, $"Invalid shape {@this.Shape} and {operand.Shape}");
            Debug.Assert(@this.Shape[0] == operand.Shape[0], $"Invalid shape {@this.Shape} and {operand.Shape}");

            for (int i = 0; i < @this.Shape[0]; i++)
                for(int j = 0; j < operand.Shape[^1]; j++)
                    operand[i, j] += @this[i];
        }
    }
}
