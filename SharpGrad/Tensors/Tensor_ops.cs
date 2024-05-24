using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public enum Operation : int
    {
        Add,
        Sub,
        Mul,
        Div,
        Store,
    }

    public struct Operation_
    {
        public readonly Operation Operation;
        public int Left;
        public int Right;
        public int Result;
    }

    public static class Operation<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static TType Add(TType left, TType right) => left + right;
        public static (TType Left, TType Right) AddBackward(TType grad, TType right, TType left) => (grad, grad);

        public static TType Sub(TType left, TType right) => left - right;
        public static TType NegBackward(TType grad, TType right, TType left) => -grad;
        public static (TType Left, TType Right) SubBackward(TType grad, TType right, TType left) => (grad, -grad);

        public static TType Mul(TType left, TType right) => left * right;
        public static (TType Left, TType Right) MulBackward(TType grad, TType right, TType left) => (grad * right, grad * left);

        public static TType Div(TType left, TType right) => left / right;
        public static (TType Left, TType Right) DivBackward(TType grad, TType right, TType left) => (grad / right, -grad * left / (right * right));

    }

    public static class Kernels<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {

        public static void Add(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = Operation<TType>.Add(left[idx], right[idx]);
        public static void AddBackward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad)
        {
            var (l, r) = Operation<TType>.AddBackward(grad[idx], right[idx], left[idx]);
            leftGrad[idx] = l;
            rightGrad[idx] = r;
        }

        public static void Sub(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = Operation<TType>.Sub(left[idx], right[idx]);

        public static void SubBackward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad)
        {
            var (l, r) = Operation<TType>.SubBackward(grad[idx], right[idx], left[idx]);
            leftGrad[idx] = l;
            rightGrad[idx] = r;
        }

        public static void Mul(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
                => output[idx] = Operation<TType>.Mul(left[idx], right[idx]);

        public static void MulBackward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad)
        {
            var (l, r) = Operation<TType>.MulBackward(grad[idx], right[idx], left[idx]);
            leftGrad[idx] = l;
            rightGrad[idx] = r;
        }

        public static void Div(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = Operation<TType>.Div(left[idx], right[idx]);

        public static void DivBackward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad)
        {
            var (l, r) = Operation<TType>.DivBackward(grad[idx], right[idx], left[idx]);
            leftGrad[idx] = l;
            rightGrad[idx] = r;
        }

        private static void KPU(Operation operation, ref TType left, ref TType right, ref TType result)
        {
            switch (operation)
            {
                case Operation.Add: result += Operation<TType>.Add(left, right); break;
                case Operation.Sub: result += Operation<TType>.Sub(left, right); break;
                case Operation.Mul: result += Operation<TType>.Mul(left, right); break;
                case Operation.Div: result += Operation<TType>.Div(left, right); break;
                case Operation.Store: result = left; break;
                default: result = TType.Zero; break;
            }
        }
        public static void Dynamic(Index1D idx, ArrayView<Operation> ops, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
        {
            for (int i = 0; i < ops.Length; i++)
                KPU(ops[i], ref left[idx], ref right[idx], ref output[idx]);
        }


        public static void KPU(Index1D idx, ArrayView<Operation_> ops, ArrayView2D<TType, Stride2D.DenseX> tensors)
        {
            TType accumulator = TType.Zero;
            for (int i = 0; i < ops.Length; i++)
            {
                Operation_ op = ops[i];
                TType left = op.Left == -1 ? accumulator : tensors[op.Left, idx];
                TType right = op.Right == -1 ? accumulator : tensors[op.Right, idx];
                TType result = op.Result == -1 ? accumulator : tensors[op.Result, idx];
                KPU(op.Operation, ref left, ref right, ref result);
                if(op.Result == -1)
                    accumulator = result;
                else
                    tensors[op.Result, idx] = result;
            }
        }
    }

    public readonly partial struct Tensor<TType> :
        IAdditionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        ISubtractionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IMultiplyOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IDivisionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static void ExecGpu(
            Operation_[] operations, List<Tensor<TType>> tensors)
        {
            MemoryBuffer1D<Operation_, Stride1D.Dense> opsOnDevice = Tensors.Accelerator.Allocate1D(operations);
            MemoryBuffer2D<TType, Stride2D.DenseX> tensorsOnDevoce = Tensors.Accelerator.Allocate2DDenseX<TType>(new LongIndex2D(tensors.Count, tensors[0].Shape.Size));
            tensors.Select(t => Tensors.Accelerator.Allocate1D(t.data)).ToList();

            Action<Index1D, ArrayView<Operation_>, ArrayView2D<TType, Stride2D.DenseX>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Operation_>, ArrayView2D<TType, Stride2D.DenseX>>(Kernels<TType>.KPU);
            loadedKernel(tensors[0].data.Length, opsOnDevice.View, tensorsOnDevoce);
            Tensors.Accelerator.Synchronize();

        }
        public static void ExecGpu(
            Operation[] operations,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            MemoryBuffer1D<Operation, Stride1D.Dense> opsOnDevice = Tensors.Accelerator.Allocate1D(operations);
            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = Tensors.Accelerator.Allocate1D(left.data);
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = Tensors.Accelerator.Allocate1D(right.data);
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = Tensors.Accelerator.Allocate1D(result.data);

            Action<Index1D, ArrayView<Operation>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Operation>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>>(Kernels<TType>.Dynamic);
            loadedKernel(left.data.Length, opsOnDevice.View, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            Tensors.Accelerator.Synchronize();

            resultOnDevice.CopyToCPU(result.data);
        }
        public static Tensor<TType> ExecGpu(Operation[] operations,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");

            var result = new Tensor<TType>(left.shape);
            ExecGpu(operations, left, right, result);
            return result;
        }

        public static void ExecGpu(
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = Tensors.Accelerator.Allocate1D(left.data);
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = Tensors.Accelerator.Allocate1D(right.data);
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = Tensors.Accelerator.Allocate1D(result.data);

            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.data.Length, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            Tensors.Accelerator.Synchronize();

            resultOnDevice.CopyToCPU(result.data);
        }

        public static Tensor<TType> ExecGpu(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");

            var result = new Tensor<TType>(left.shape);
            ExecGpu(func, left, right, result);
            return result;
        }

        public static Tensor<TType> operator +(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernels<TType>.Add, left, right);

        public static Tensor<TType> operator -(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernels<TType>.Sub, left, right);

        public static Tensor<TType> operator *(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernels<TType>.Mul, left, right);

        public static Tensor<TType> operator /(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernels<TType>.Div, left, right);
    }
}
