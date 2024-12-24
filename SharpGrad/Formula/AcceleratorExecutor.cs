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
        private class TensorCoordinateMapper<TShape, TIndices, TXD, T>(
            ArrayView1D<T, Stride1D.Dense> inputData,
            ArrayView1D<T, Stride1D.Dense> outputData,
            ArrayView1D<InternalDimension, Stride1D.Dense> dimensions,
            ArrayView1D<TShape, Stride1D.Dense> shapes,
            ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> datas,
            ArrayView1D<InternalOperation<T>, Stride1D.Dense> operations,
            T[] reults,
            T[] gradients)
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            public readonly ArrayView1D<T, Stride1D.Dense> InputData = inputData;
            public readonly ArrayView1D<T, Stride1D.Dense> OutputData = outputData;
            public readonly ArrayView1D<InternalDimension, Stride1D.Dense> Dimensions = dimensions;
            public readonly ArrayView1D<TShape, Stride1D.Dense> Shapes = shapes;
            public readonly ArrayView1D<InternalTensor<TShape, TIndices, TXD>, Stride1D.Dense> Tensors = datas;
            public readonly ArrayView1D<InternalOperation<T>, Stride1D.Dense> Operations = operations;
            public readonly T[] Results = reults;
            public readonly T[] Gradients = gradients;

            private TIndices GetDimensionsSize(TShape shape)
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

            public bool GetShape(BIndex<byte> shapeIdx, out AcceleratorShape<TShape, TIndices, TXD> acceleratorShape)
            {
                if (shapeIdx.IsEmpty)
                {
                    acceleratorShape = default;
                    return false;
                }
                else
                {
                    acceleratorShape = new AcceleratorShape<TShape, TIndices, TXD>(Shapes, Dimensions, shapeIdx);
                    return true;
                }
            }

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
            private TIndices ComputeIndices(TShape shape, long flatIndex)
                => ComputeIndices(GetDimensionsSize(shape), flatIndex);

            private bool ProjectIndices_(AcceleratorShapeIndices<TShape, TIndices, TXD> from, AcceleratorShape<TShape, TIndices, TXD> to, out AcceleratorShapeIndices<TShape, TIndices, TXD> projectedIndices)
            {
                TIndices indicesTo = default;
                for (int d = 0; d < to.DimsSize.Count; d++)
                {
                    BIndex<byte> iDim = to.DimsIndex[d];
                    if (!iDim.IsEmpty)
                    {
                        int iFromIndices = from.Shape.DimsIndex.IndexOf(iDim);
                        if (iFromIndices < 0)
                        {
                            projectedIndices = default;
                            return false;
                        }

                        int fromIndice = from.Indices[iFromIndices];
                        if (fromIndice < 0 || fromIndice >= Dimensions[iDim].Size)
                        {
                            projectedIndices = default;
                            return false;
                        }
                        indicesTo[d] = fromIndice;
                    }
                }

                projectedIndices = new AcceleratorShapeIndices<TShape, TIndices, TXD>(to, indicesTo);
                return true;
            }
            private bool ProjectIndices(AcceleratorShapeIndices<TShape, TIndices, TXD> from, AcceleratorShape<TShape, TIndices, TXD> to, out AcceleratorShapeIndices<TShape, TIndices, TXD> projectedIndices)
            {
                if (from.Shape.Idx == to.Idx)
                {
                    projectedIndices = from;
                    return true;
                }
                if (from.Shape.Idx.IsEmpty || to.Idx.IsEmpty)
                {
                    projectedIndices = default;
                    return to.Idx.IsEmpty;
                }

                return ProjectIndices_(from, to, out projectedIndices);
            }

            private bool ProjectIndices(AcceleratorShapeIndices<TShape, TIndices, TXD> from, BIndex<byte> toShapeIdx, out AcceleratorShapeIndices<TShape, TIndices, TXD> projectedIndices)
            {
                if (from.Shape.Idx == toShapeIdx)
                {
                    projectedIndices = from;
                    return true;
                }
                if (from.Shape.Idx.IsEmpty || toShapeIdx.IsEmpty)
                {
                    projectedIndices = default;
                    return toShapeIdx.IsEmpty;
                }
                if (GetShape(toShapeIdx, out AcceleratorShape<TShape, TIndices, TXD> to))
                    return ProjectIndices_(from, to, out projectedIndices);
                else
                {
                    projectedIndices = default;
                    return false;
                }
            }

            private long ComputeFlatIndex(AcceleratorShapeIndices<TShape, TIndices, TXD> acceleratorShapeIndices)
            {
                long flatIndex = 0;
                if (!acceleratorShapeIndices.Shape.Idx.IsEmpty)
                {
                    for (int d = 0; d < acceleratorShapeIndices.Shape.DimsIndex.Count; d++)
                    {
                        BIndex<byte> iDim = acceleratorShapeIndices.Shape.DimsIndex[d];
                        if (!iDim.IsEmpty)
                        {
                            int iFromIndices = acceleratorShapeIndices.Shape.DimsIndex.IndexOf(iDim);
                            flatIndex *= acceleratorShapeIndices.Shape.DimsSize[iFromIndices];
                            flatIndex += acceleratorShapeIndices.Indices[iFromIndices];
                        }
                    }
                }
                return flatIndex;
            }
            private long ComputeFlatIndex(TShape shape, TIndices indices)
            {
                long index = 0;
                for (int d = 0; d < shape.Count; d++)
                {
                    BIndex<byte> iDim = shape[d];
                    if (!iDim.IsEmpty)
                    {
                        index *= Dimensions[iDim].Size;
                        index += indices[d];
                    }
                }
                return index;
            }

            private TIndices ComputeIndicesOnly(TShape shape, long flatIdx)
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

            private T GetValue(InternalTensor<TShape, TIndices, TXD> tensor, long flatIndex = 0)
                => tensor.Source == SourceOfOperand.Input
                ? InputData[tensor.Offset + flatIndex]
                : OutputData[tensor.Offset + flatIndex];
            private bool GetValue(InternalTensor<TShape, TIndices, TXD> tensor, AcceleratorShapeIndices<TShape, TIndices, TXD> indices, out T value)
            {
                if (ProjectIndices(indices, tensor.ShapeIdx, out AcceleratorShapeIndices<TShape, TIndices, TXD> tensorIndices))
                {
                    value = GetValue(tensor, ComputeFlatIndex(tensorIndices));
                    return true;
                }
                else
                {
                    value = default;
                    return false;
                }
            }
            public bool GetValue(OperandIndex<sbyte> operation, AcceleratorShapeIndices<TShape, TIndices, TXD> indices, out T value)
            {
                if (operation.IsOperation)
                {
                    value = Results[operation.Index];
                    return true;
                }
                else if (operation.IsTensor)
                {
                    return GetValue(Tensors[operation.Index], indices, out value);
                }
                else
                {
                    value = default;
                    return false;
                }
            }

            private bool SetValue(InternalTensor<TShape, TIndices, TXD> tensor, T value, long flatIndex)
            {
                if (tensor.Source == SourceOfOperand.Output)
                {
                    flatIndex = tensor.ShapeIdx.IsEmpty
                        ? tensor.Offset
                        : tensor.Offset + flatIndex;
                    OutputData[flatIndex] = value;
                    return true;
                }
                return false;
            }
            private bool SetValue(InternalTensor<TShape, TIndices, TXD> tensor, T value, AcceleratorShapeIndices<TShape, TIndices, TXD> indices)
            {
                if (tensor.Source == SourceOfOperand.Output)
                {
                    OutputData[tensor.Offset + ComputeFlatIndex(indices)] = value;
                    return true;
                }
                else
                {
                    return false;
                }
            }

            public bool SetValue(BIndex<byte> outputTensorIdx, T value, AcceleratorShapeIndices<TShape, TIndices, TXD> indices)
                => !outputTensorIdx.IsEmpty && SetValue(Tensors[outputTensorIdx], value, indices);

            public InternalOperation<T> FinalOperation => Operations[Operations.IntExtent.X - 1];

            public bool GetBeforeReductionShapeIdx(out BIndex<byte> beforeResductionShapeIdx)
            {
                InternalOperation<T> final = FinalOperation;
                if (final.IsReduction)
                {
                    beforeResductionShapeIdx = final.RightIdx.IsOperation
                        ? Operations[final.RightIdx.Index].ShapeIdx
                        : Tensors[final.RightIdx.Index].ShapeIdx;
                    return true;
                }
                beforeResductionShapeIdx = default;
                return false;
            }

            internal bool SetResult(byte operationIdx, T result, AcceleratorShapeIndices<TShape, TIndices, TXD> indices)
            {
                Results[operationIdx] = result;
                InternalOperation<T> operation = Operations[operationIdx];
                return !operation.IsOutput || SetValue(operation.OutputIdx, result, indices);
            }

            internal bool GetReductionDimIdx(out BIndex<byte> dimToReduceIdx)
            {
                InternalOperation<T> finalOperation = FinalOperation;
                if (finalOperation.OpCode.HasFlag(OpCode.IsReduction)
                    && !finalOperation.LeftIdx.IsEmpty)
                {
                    dimToReduceIdx = (byte)finalOperation.LeftIdx.Index;
                    return true;
                }

                dimToReduceIdx = BIndex<byte>.Empty;
                return false;
            }

            internal bool GetBaseIndices(LongIndex1D idx, out AcceleratorShapeIndices<TShape, TIndices, TXD> baseIndices)
            {
                InternalOperation<T> finalOperation = FinalOperation;

                BIndex<byte> shapeIdx;
                if (finalOperation.IsReduction)
                {
                    shapeIdx = finalOperation.RightIdx.IsOperation
                        ? Operations[finalOperation.RightIdx.Index].ShapeIdx
                        : Tensors[finalOperation.RightIdx.Index].ShapeIdx;
                }
                else
                {
                    shapeIdx = finalOperation.ShapeIdx;
                }
                if (GetShape(shapeIdx, out AcceleratorShape<TShape, TIndices, TXD> acceleratorShape))
                {
                    TIndices indices = ComputeIndices(acceleratorShape.DimsSize, idx);
                    baseIndices = new AcceleratorShapeIndices<TShape, TIndices, TXD>(acceleratorShape, indices);
                    return true;
                }
                else
                {
                    baseIndices = default;
                    return false;
                }
            }
        }

        public static readonly Context context = Context.Create(builder => builder.AllAccelerators());
        public static readonly Device device = context.GetPreferredDevice(preferCPU: false);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

        private static bool OperationInvoke<T>(OpCode opCode, T left, out T result)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            switch (opCode)
            {
                case OpCode.Neg:
                    result = NegOp<T>.Invoke(left);
                    return true;
                case OpCode.Log:
                    result = LogOp<T>.Invoke(left);
                    return true;
                case OpCode.Exp:
                    result = ExpOp<T>.Invoke(left);
                    return true;
            }

            result = T.Zero;
            return false;
        }

        private static bool ReduceInvoke<T>(OpCode opCode, T left, T right, out T result)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            switch (opCode)
            {
                case OpCode.Sum:
                    result = AddOp<T>.Invoke(left, right); // TODO: This should use SumOp instead
                    return true;
            }
            result = T.Zero;
            return false;
        }

        private static bool OperationInvoke<T>(OpCode opCode, T left, T right, out T result)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            switch (opCode)
            {
                case OpCode.Add:
                    result = AddOp<T>.Invoke(left, right);
                    return true;
                case OpCode.Sub:
                    result = SubOp<T>.Invoke(left, right);
                    return true;
                case OpCode.Mul:
                    result = MulOp<T>.Invoke(left, right);
                    return true;
                case OpCode.Div:
                    result = DivOp<T>.Invoke(left, right);
                    return true;
                case OpCode.Pow:
                    result = PowOp<T>.Invoke(left, right);
                    return true;
            }
            result = T.Zero;
            return false;
        }


        private static bool ForwardAtOperationIdx<TShape, TIndices, TXD, T>(
            TensorCoordinateMapper<TShape, TIndices, TXD, T> kernelContext,
            AcceleratorShapeIndices<TShape, TIndices, TXD> indices,
            byte operationIdx
        )
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            InternalOperation<T> operation = kernelContext.Operations[operationIdx];
            if (kernelContext.GetValue(operation.LeftIdx, indices, out T left))
            {
                if (operation.RightIdx.IsEmpty)
                {
                    if (OperationInvoke(operation.OpCode, left, out T value))
                    {
                        return kernelContext.SetResult(operationIdx, value, indices);
                    }
                }
                else
                {
                    if (operation.OpCode.HasFlag(OpCode.IsReduction))
                    {
                        if (kernelContext.GetValue(operation.RightIdx, indices, out T right)
                            && ReduceInvoke(operation.OpCode, kernelContext.Results[operationIdx], right, out T result))
                        {
                            kernelContext.Results[operationIdx] = result;
                            return true;
                        }
                    }
                    else if (kernelContext.GetValue(operation.RightIdx, indices, out T right)
                        && OperationInvoke(operation.OpCode, left, right, out T result)
                        && operation.IsOutput)
                    {
                        return kernelContext.SetResult(operationIdx, result, indices);
                    }
                }
            }
            return false;
        }

        private static bool ForwardOperations<TShape, TIndices, TXD, T>(
            TensorCoordinateMapper<TShape, TIndices, TXD, T> kernelContext,
            AcceleratorShapeIndices<TShape, TIndices, TXD> indices)
            where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
            where TIndices : unmanaged, IInternalStaticArray<int, TXD>
            where TXD : struct, IXD
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            byte count = (byte)kernelContext.Operations.IntExtent.X;
            for (byte i = 0; i < count; i++)
            {
                if (!ForwardAtOperationIdx(kernelContext, indices, i))
                    return false;
            }
            return true;
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

            TensorCoordinateMapper<TShape, TIndices, TXD, T> kernelContext = new(inputData, outputData, dimensions, shapes, datas, operations, new T[operationCount], new T[operationCount]);

            InternalOperation<T> lastOperation = kernelContext.FinalOperation;
            kernelContext.GetBaseIndices(idx, out AcceleratorShapeIndices<TShape, TIndices, TXD> indices);
            int dimToReduceSize;
            if (kernelContext.GetReductionDimIdx(out BIndex<byte> dimToReduceIdx))
            {
                int iDim = indices.Shape.DimsIndex.IndexOf(dimToReduceIdx);
                if (indices.Indices[iDim] != 0)
                    // TODO: Compute starting indice of the dimension to reduce should be 0. Some error should be thrown
                    return;

                dimToReduceSize = kernelContext.Dimensions[dimToReduceIdx].Size;

                // Use a warp to reduce the dimension if it is as large as the warp size.
                // Use a single thread to reduce the dimension if it is smaller than the warp size.
                int step = dimToReduceSize < Warp.WarpSize ? 1 : Warp.WarpSize;
                for (int i = dimToReduceSize < Warp.WarpSize ? 1 : Warp.LaneIdx; i < dimToReduceSize; i += step)
                {
                    indices.Indices[iDim] = i;
                    ForwardOperations(kernelContext, indices);
                }

                // If a warp was used to reduce the dimension, reduce the results of the warp.
                if (step == Warp.WarpSize)
                {
                    T value = kernelContext.Results[^1];
                    for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
                    {
                        value += Warp.ShuffleDown(value, offset);
                    }
                    if (Warp.IsFirstLane)
                    {
                        kernelContext.Results[^1] = value;
                    }
                }

                // Set the result of the reduction to the output tensor
                kernelContext.SetValue(lastOperation.OutputIdx, kernelContext.Results[^1], indices);
            }
            else
            {
                ForwardOperations(kernelContext, indices);
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
