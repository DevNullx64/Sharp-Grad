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

        private readonly struct TensorCoordinateMapper<TShape, TIndices, TXD, T>(
            ArrayView1D<T, Stride1D.Dense> inputData,
            ArrayView1D<T, Stride1D.Dense> outputData,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            ArrayView1D<TShape, Stride1D.Dense> shapes,
            ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> datas,
            ArrayView1D<InternalOperation<T>, Stride1D.Dense> operations)
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            public readonly ArrayView1D<T, Stride1D.Dense> InputData = inputData;
            public readonly ArrayView1D<T, Stride1D.Dense> OutputData = outputData;
            public readonly ArrayView1D<InternalDimension, Stride1D.Dense> Dimensions = dimensions;
            public readonly ArrayView1D<TShape, Stride1D.Dense> Shapes = shapes;
            public readonly ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> Datas = datas;
            public readonly ArrayView1D<InternalOperation<T>, Stride1D.Dense> Operations = operations;

            public static TIndices ComputeIndices(TIndices dimensionSizes, long flatIndex)
            {
                TIndices indices = default;
                for (int d = 0; d < dimensionSizes.Count; d++)
                {
                    int size = dimensionSizes[d];
                    if (size > 1)
                    {
                        indices[d] = (int)(flatIndex % size);
                        flatIndex /= size;
                    }
                }
                return indices;
            }

            public readonly long ComputeFlatIndex(BIndex<byte> shapeIdx, TIndices currentIndices)
            {
                if (shapeIdx.IsEmpty)
                    return default;
                TShape shape = Shapes[shapeIdx];
                long index = 0;
                for (int d = 0; d < shape.Count; d++)
                {
                    BIndex<byte> iDim = shape[d];
                    if (!iDim.IsEmpty)
                    {
                        index *= Dimensions[iDim].Size;
                        index += currentIndices[d];
                    }
                }
                return index;
            }

            public readonly TIndices GetDimensionsSize(TShape shape)
            {
                TIndices dimsSize = default;
                for (int d = 0; d < shape.Count; d++)
                {
                    BIndex<byte> iDim = shape[d];
                    if (iDim.IsEmpty)
                    {
                        dimsSize[d] = 1;
                    }
                    else
                    {
                        dimsSize[d] = Dimensions[iDim].Size;
                    }
                }
                return dimsSize;
            }

            public readonly TIndices ComputeIndicesOnly(TShape shape, long flatIdx)
            {
                TIndices indices = default;
                for (int d = shape.Count - 1; d >= 0; d--)
                {
                    BIndex<byte> iDim = shape[d];
                    if (iDim.IsEmpty)
                    {
                        indices[d] = 0;
                    }
                    else
                    {
                        InternalDimension dim = Dimensions[iDim];
                        indices[d] = (int)(flatIdx % dim.Size);
                        flatIdx /= dim.Size;
                    }
                }
                if (flatIdx > 0)
                    return default;
                return indices;
            }

            public readonly InternalShapeIndices<TShape, TIndices, TXD> ComputeShapeIndices(BIndex<byte> shapeIdx, long flatIndex)
            {
                TShape shape = Shapes[shapeIdx];
                TIndices indices = ComputeIndicesOnly(shape, flatIndex);
                return new InternalShapeIndices<TShape, TIndices, TXD>(shape, indices);
            }
        }

        private static void Forward<TShape, TIndices, TXD, T>(
            TensorCoordinateMapper<TShape, TIndices, TXD, T> decoder,
            LongIndex1D idx,
            ArrayView1D<InternalOperation<T>, Stride1D.Dense> operations,
            SpecializedValue<byte> operationCount
            )
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {

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
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            byte opCount = (byte)operations.IntExtent.X;
            if (operationCount != opCount)
                return;

            TensorCoordinateMapper<TShape, TIndices, TXD, T> decoder = new(inputData, outputData, dimensions, shapes, datas, operations);

            InternalOperation<T> last = operations[opCount - 1];
            TShape beforeShape;
            sbyte dimToReduceIdx;
            TShape outShape = shapes[last.ShapeIdx];
            TIndices outSizes = decoder.GetDimensionsSize(outShape);
            int dimToReduceSize;
            TIndices currentIndices;

            if (last.OpCode.HasFlag(OpCode.IsReduction))
            {
                opCount--; // Exclude reduction operation
                BIndex<byte> beforeShapeIdx = last.LeftIdx.IsOperation
                    ? operations[last.LeftIdx.Index].ShapeIdx
                    : datas[last.LeftIdx.Index].ShapeIdx;
                InternalShape<TShape, TIndices, TXD> beforeFullShape = new(shapes, dimensions, beforeShapeIdx);
                beforeShape = beforeFullShape.Shape;

                dimToReduceIdx = last.RightIdx.Index;
                int iDimToReduce = beforeFullShape.Shape.IndexOf((byte)dimToReduceIdx);
                dimToReduceSize = beforeFullShape.Sizes[iDimToReduce];

                TIndices laneIndices = beforeFullShape.Sizes;
                laneIndices[iDimToReduce] = dimToReduceSize < Warp.WarpSize ? 1 : Warp.WarpSize;
                currentIndices = TensorCoordinateMapper<TShape, TIndices, TXD, T>.ComputeIndices(laneIndices, idx);
            }
            else
            {
                beforeShape = outShape;
                dimToReduceIdx = -1;
                dimToReduceSize = 1;
                currentIndices = TensorCoordinateMapper<TShape, TIndices, TXD, T>.ComputeIndices(outSizes, idx);
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
                        if(leftTensor.ShapeIdx == op.ShapeIdx)
                        {
                            left = inputData[decoder.ComputeFlatIndex(leftTensor.ShapeIdx, currentIndices)];
                        }
                        else
                        {
                            left = valCache[op.LeftIdx.Index];
                        }
                    }
                }
                currentIndices[dimToReduceIdx] += Warp.WarpSize;
            }

            if (dimToReduceSize >= Warp.WarpSize)
            {
                // Compute inter lane reduction
            }
        }

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
