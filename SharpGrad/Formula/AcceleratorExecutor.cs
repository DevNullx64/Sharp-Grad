using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Formula.Internal;
using SharpGrad.Operators;
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace SharpGrad.Formula
{
    public static class AcceleratorExecutor
    {
        public static readonly Context context = Context.Create(builder => builder.AllAccelerators());
        public static readonly Device device = context.GetPreferredDevice(preferCPU: false);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

        private static T OperationInvoke<T>(OpCode opCode, T left)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Neg => NegOp<T>.Invoke(left),
                OpCode.Log => LogOp<T>.Invoke(left),
                OpCode.Exp => ExpOp<T>.Invoke(left),
                //OpCode.Sqrt => SqrtOp<TCoordinates>.Invoke(left),
                //OpCode.Sin => SinOp<TCoordinates>.Invoke(left),
                //OpCode.Cos => CosOp<TCoordinates>.Invoke(left),
                //OpCode.Tan => TanOp<TCoordinates>.Invoke(left),
                _ => default // TODO: This should not happen
            };

        private static T OperationInvoke<T>(OpCode opCode, T left, T right)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Add => AddOp<T>.Invoke(left, right),
                OpCode.Sub => SubOp<T>.Invoke(left, right),
                OpCode.Mul => MulOp<T>.Invoke(left, right),
                OpCode.Div => DivOp<T>.Invoke(left, right),
                OpCode.Pow => PowOp<T>.Invoke(left, right),
                _ => default // TODO: This should not happen
            };

        private static T Forward<TShape, TIndices, TXD, T>(
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            ArrayView1D<TShape, Stride1D.Dense> shapes,
            ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> datas,
            Index1D idx,
            InternalOperation<T> op
            )
            where TShape : unmanaged, IInternalShape<TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            InternalTensor<TShape, TIndices, TXD> leftTensor = datas[op.LeftIdx];
            if(op.ShapeIdx == leftTensor.ShapeIdx)
            {

            }
            else
            {

            }
        }

        private static TIndices GetDimensionsSize<TShape, TIndices, TXD>(
            TShape shape,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions
        )
            where TShape : unmanaged, IInternalShape<TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
        {
            TIndices dimsSize = default;
            for (int d = 0; d < shape.Rank; d++)
            {
                dimsSize[d] = dimensions[shape[d]].Size;
            }
            return dimsSize;
        }

        private static TIndices GetIndicesOnly<TShape, TIndices, TXD>(
            TShape shape,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            long flatIdx
        )
            where TShape : unmanaged, IInternalShape<TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
        {
            TIndices indices = default;
            for (int d = shape.Rank - 1; d >= 0; d--)
            {
                InternalDimension dim = dimensions[shape[d]];
                indices[d] = (int)(flatIdx % dim.Size);
                flatIdx /= dim.Size;
            }
            if (flatIdx > 0)
                return default;
            return indices;
        }

        private static (TIndices DimsSize, TIndices Indices) GetIndices<TShape, TIndices, TXD>(
            TShape shape,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            long flatIdx
        )
            where TShape : unmanaged, IInternalShape<TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
        {
            TIndices dimsSize = default;
            TIndices indices = default;
            for (int d = shape.Rank - 1; d >= 0; d--)
            {
                InternalDimension dim = dimensions[shape[d]];
                dimsSize[d] = dim.Size;
                indices[d] = (int)(flatIdx % dim.Size);
                flatIdx /= dim.Size;
            }
            if(flatIdx > 0)
                return (default, default);
            return (dimsSize, indices);
        }

        private static void Forward<TShape, TIndices, TXD, T>(
            LongIndex1D idx,
            ArrayView1D<T, Stride1D.Dense> inputData,
            ArrayView1D<T, Stride1D.Dense> outputData,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            ArrayView1D<TShape, Stride1D.Dense> shapes,
            ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> datas,
            ArrayView1D<InternalOperation<T>, Stride1D.Dense> operations,
            SpecializedValue<byte> operationCount
        )
            where TShape : unmanaged, IInternalShape<TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            byte opCount = (byte)operations.IntExtent.X;
            if (operationCount != opCount)
                return;

            InternalOperation<T> last = operations[opCount - 1];
            TShape beforeShape;
            sbyte dimToReduceIdx;
            TShape outShape = shapes[last.ShapeIdx];
            TIndices outSizes = GetDimensionsSize<TShape, TIndices, TXD>(outShape, dimensions);
            int dimToReduceSize;
            TIndices currentIndices;

            if (last.OpCode.HasFlag(OpCode.IsReduction))
            {
                opCount--; // Exclude reduction operation
                BIndex<byte> beforeShapeIdx = last.LeftIdx.IsOperation
                    ? operations[last.LeftIdx.Index].ShapeIdx
                    : datas[last.LeftIdx.Index].ShapeIdx;
                InternalFullShape<TShape, TIndices, TXD> beforeFullShape = new(shapes, dimensions, beforeShapeIdx);
                beforeShape = beforeFullShape.Shape;

                dimToReduceIdx = last.RightIdx.Index;
                int iDimToReduce = beforeFullShape.Shape.IndexOf((byte)dimToReduceIdx);
                dimToReduceSize = beforeFullShape.Sizes[iDimToReduce];

                TIndices laneIndices = beforeFullShape.Sizes;
                laneIndices[iDimToReduce] = dimToReduceSize < Warp.WarpSize ? 1 : Warp.WarpSize;
                currentIndices = TShape.GetIndices(laneIndices, idx);
            }
            else
            {
                beforeShape = outShape;
                dimToReduceIdx = -1;
                dimToReduceSize = 1;
                currentIndices = TShape.GetIndices(outSizes, idx);
            }

            T[] valCache = new T[opCount];
            T[] gradCache = new T[opCount];

            currentIndices[dimToReduceIdx] = dimToReduceSize < Warp.WarpSize ? 1 : Warp.LaneIdx;
            while (currentIndices[dimToReduceIdx] < dimToReduceSize)
            {
                for (int i = 0; i < opCount; i++)
                {
                    InternalOperation<T> op = operations[i];
                    T val = default;
                    T grad = default;
                    T left;
                    if (op.LeftIdx.IsOperation)
                    {
                        left = valCache[op.LeftIdx.Index];
                    }
                    else
                    {
                        InternalTensor<TShape, TIndices, TXD> leftTensor = datas[op.LeftIdx.Index];

                    }
                }
                currentIndices[dimToReduceIdx] += Warp.WarpSize;
            }

            if (dimToReduceSize >= Warp.WarpSize)
            {
                // Compute inter lane reduction
            }
        }

        private static T Forward<T>(
            Index1D idx,
            InternalOperation<T> op,
            ArrayView2D<T, Stride2D.DenseY> datas,
            T[] cache)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T left = op.LeftIdx.Category == SourceOfOperand.Data
                ? datas[op.LeftIdx.Index, idx]
                : cache[op.LeftIdx.Index];

            if (op.RightIdx.IsEmpty)
            {
                return OperationInvoke(op.OpCode, left);
            }
            else
            {
                T right = op.RightIdx.Category == SourceOfOperand.Data
                    ? datas[op.RightIdx.Index, idx]
                    : cache[op.RightIdx.Index];
                return OperationInvoke(op.OpCode, left, right);
            }
        }

        private static void ForwardKernel<T>(
            Index1D idx,
            ArrayView<InternalOperation<T>> ops,
            ArrayView2D<T, Stride2D.DenseY> datas,
            ArrayView2D<T, Stride2D.DenseY> outputs,
            SpecializedValue<ushort> cacheSize)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T[] cache = new T[cacheSize];
            for (int i = 0; i < cacheSize; i++)
            {
                var op = ops[i];
                cache[op.OutputIdx] = Forward(idx, op, datas, cache);
                if (!op.OutputIdx.IsEmpty)
                    outputs[(int)op.OutputIdx, idx] = cache[op.OutputIdx];
            }
        }

        /*
        internal static void Forward<TCoordinates>(
            SafeAccelerator accelerator,
            OperationInfo<TCoordinates>[] ops,
            AcceleratorBuffer<TCoordinates>[] datas,
            out AcceleratorBuffer<TCoordinates>[] outputs)
            where TCoordinates : unmanaged, INumber<TCoordinates>, IExponentialFunctions<TCoordinates>, ILogarithmicFunctions<TCoordinates>, IPowerFunctions<TCoordinates>
        {
            if(datas.Any(d => d.SafeAccelerator != accelerator))
                throw new ArgumentException("All data buffers must be on the same accelerator.");


        }
        */

        private static T Backward<T>(OpCode opCode, T left, T currentGrad)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Neg => NegOp<T>.Backward(left, currentGrad),
                OpCode.Log => LogOp<T>.Backward(left, currentGrad),
                OpCode.Exp => ExpOp<T>.Backward(left, currentGrad),
                _ => default // TODO: This should not happen
            };

        private static (T leftGrad, T rightGrad) Backward<T>(OpCode opCode, T left, T right, T currentGrad)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Add => AddOp<T>.Backward(left, right, currentGrad),
                OpCode.Sub => SubOp<T>.Backward(left, right, currentGrad),
                OpCode.Mul => MulOp<T>.Backward(left, right, currentGrad),
                OpCode.Div => DivOp<T>.Backward(left, right, currentGrad),
                OpCode.Pow => PowOp<T>.Backward(left, right, currentGrad),
                _ => default // TODO: This should not happen
            };

        private static void BackwardKernel<T>(
            Index1D idx,
            ArrayView<InternalOperation<T>> ops,
            ArrayView2D<T, Stride2D.DenseY> datas,
            ArrayView1D<BIndex<ushort>, Stride1D.Dense> dataGradIndices,
            ArrayView1D<T, Stride1D.Dense> outputs,
            ArrayView1D<T, Stride1D.Dense> grads,
            SpecializedValue<ushort> cacheSize)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T[] cache = new T[cacheSize];
            T[] gradCache = new T[cacheSize];

            // Forward pass
            for (int i = 0; i < ops.Length; i++)
            {
                var op = ops[i];
                cache[op.OutputIdx] = Forward(idx, op, datas, cache);
                if (!op.OutputIdx.IsEmpty)
                    outputs[op.OutputIdx] = cache[op.OutputIdx];
                gradCache[i] = !op.GradientIndex.IsEmpty
                    ? grads[op.GradientIndex]
                    : T.Zero;
            }

            // Backward pass
            for (int i = cacheSize - 1; i >= 0; i--)
            {
                var op = ops[i];
                T currentGrad = gradCache[i];

                T left = op.LeftIdx.Category == SourceOfOperand.Data
                    ? datas[idx, op.LeftIdx.Index]
                    : cache[op.LeftIdx.Index];

                T leftGrad;
                if (op.RightIdx.IsEmpty)
                {
                    leftGrad = Backward(op.OpCode, left, currentGrad);
                }
                else
                {
                    T right = op.RightIdx.Category == SourceOfOperand.Data
                        ? datas[idx, op.RightIdx.Index]
                        : cache[op.RightIdx.Index];

                    (leftGrad, T rightGrad) = Backward(op.OpCode, left, right, currentGrad);

                    if (op.RightIdx.Category == SourceOfOperand.Data)
                        grads[op.RightIdx.Index] += rightGrad;
                    else
                        gradCache[op.RightIdx.Index] += rightGrad;
                }

                if (op.LeftIdx.Category == SourceOfOperand.Data)
                    grads[op.LeftIdx.Index] += leftGrad;
                else
                    gradCache[op.LeftIdx.Index] += leftGrad;
            }
        }

        public static async Task ForwardAsync<T>(AcceleratorExtender accelerator, KPUScript<T> script)
        where T : unmanaged, INumber<T>
        {
            foreach (var data in script.datas)
            {
                var b = data.IsExclusiveLockHeld; // TODO: This should be async
                data.SafeAccelerator = accelerator;
            }
        }
    }
}
