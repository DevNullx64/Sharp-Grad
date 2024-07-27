using ILGPU.Runtime;
using ILGPU;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using SharpGrad.Tensors.Operators;
using System.Runtime.CompilerServices;
using System.Runtime;
using ILGPU.Runtime.Cuda;
using System.Data.SqlTypes;
using System.Data;
using System.Net.Http.Headers;
using System.Diagnostics;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        /// <summary>
        /// Execute a list of operations on tensors and save result reduced in the args.applyDim dimension.
        /// </summary>
        /// <typeparam name="T">Type of the tensor</typeparam>
        /// <typeparam name="TOp">Type of the reduction operation</typeparam>
        /// <param name="idxResult">GPU Index in result tensor</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="input">Tensors to operate on</param>
        /// <param name="result">Result tensor</param>
        /// <param name="shape">Shape of the input tensor</param>
        /// <param name="args">Additional arguments. See <see cref="ByteArgs"/></param>
        /// <remarks>The reduction is performed in the applyDim dimension.</remarks>
        private static void ReduceKernel<T, TOp>(
            Index1D idxResult, 
            ArrayView<OperationKPU> ops, 
            ArrayView2D<T, Stride2D.DenseY> input,
            ArrayView1D<T, Stride1D.Dense> result,
            ArrayView1D<int, Stride1D.Dense> shape,
            SpecializedValue<ByteArgs> args)
            where T : unmanaged, INumber<T>
            where TOp : IExecutor2<T, T, T>
        {
            //Debug.Assert(args.Value.ShapeDims == shape.Length, $"Invalid shape dims {args.Value.ShapeDims} for shape {shape.Length}");
            //Debug.Assert(args.Value.ReduceCount > 0 && args.Value.ReduceCount <= 32, $"Invalid reduce count args.Value.{args.Value.ReduceCount}");

            // First, we compute the indices of result.
            int[] indices = new int[args.Value.ShapeDims];
            int idx = idxResult;
            for (int i = shape.IntLength - 1; i >= 0; i--)
            {
                int idxApply = shape[i];
                if (i == args.Value.ApplyDim)
                    idxApply = (shape[i] + args.Value.ReduceCount - 1) / args.Value.ReduceCount;

                indices[i] = idxApply != 1 ? idx % idxApply : 0;
                idx /= idxApply;
            }

            // Then we can reuse the indices to compute the first index in the input tensor.
            indices[args.Value.ApplyDim] *= args.Value.ReduceCount; // set the base input index of the dimension to reduce.
            int idxInput = Shape.GetFlattenIndices(shape, indices); // compute the first index in the input tensor.

            // Then we compute the last index in the input tensor.
            indices[args.Value.ApplyDim] += args.Value.ReduceCount - 1; // set the last input index in the dimension to reduce.
            if (indices[args.Value.ApplyDim] >= shape[args.Value.ApplyDim]) // if the last index is greater than the dimension size,
                indices[args.Value.ApplyDim] = shape[args.Value.ApplyDim] - 1; // we set it to the dimension size.
            int lastIdxInput = Shape.GetFlattenIndices(shape, indices); // compute the last index in the input tensor.

            // Compute the offset between two elements of the applyDim dimension.
            int stepSize = 1;
            for(int i = shape.IntExtent - 2; i >= args.Value.ApplyDim; i--)
                stepSize *= shape[i + 1];

            // Compute the first element of the reduction.
            T[] cache = new T[args.Value.CacheSize]; // Cache to store intermediate results.
            T acc = ExecKernel(idxInput, ops, input, cache); // Compute the first element of the reduction.

            // Reduce the elements.
            while ((idxInput += stepSize) <= lastIdxInput) // Move to the next element of the reduction.
                acc = TOp.Exec(acc, ExecKernel(idxInput, ops, input, cache));

            result[idxResult] = acc;
        }

        private int ReduceKernelElementsCount = 32;
        private static void ReduceKernel<T, TOp>(
            Index1D idxDestination,
            ArrayView1D<T, Stride1D.Dense> source,
            ArrayView1D<int, Stride1D.Dense> sourceShape,
            ArrayView1D<T, Stride1D.Dense> destination,
            int dim,
            int count,
            SpecializedValue<int> dims)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            // Duplicate the tensor shape. Except for the dimension to reduce, where the size is divided by the count.
            int[] destinationShape = new int[dims];
            for (int i = 0; i < dims; i++)
                destinationShape[i] = (i == dim) ? (sourceShape[i] + count - 1) / count : sourceShape[i];

            // Get the indices of input and output tensors.
            int[] indicesDestination = Shape.IndicesFrom(destinationShape, idxDestination);
            int[] indicesSource = new int[dims];
            for(int i = 0; i < dims; i++)
                indicesSource[i] = (i == dim) ? indicesDestination[i] * count : indicesDestination[i];

            // Compute the amount of elements that can be reduced.
            int cMax = indicesSource[dim] + count;
            if (cMax > sourceShape[dim])
                cMax = sourceShape[dim];

            // Compute the reduction of the two first elements.
            int iSource = Shape.GetFlattenIndices(sourceShape, indicesSource);
            indicesSource[dim]++;
            int iSource2 = Shape.GetFlattenIndices(sourceShape, indicesSource);
            indicesSource[dim]++;
            T acc = TOp.Exec(source[iSource], source[iSource2]);

            // Reduce the elements.
            while (indicesSource[dim] < cMax)
            {
                iSource2 = Shape.GetFlattenIndices(sourceShape, indicesSource);
                acc = TOp.Exec(acc, source[iSource2]);
                indicesSource[dim]++;
            }

            // Store the result.
            destination[idxDestination] = acc;
        }

        public TensorData<T> Reduce<T, TOp>(Tensor<T> tensor, Index? dim = null)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            // If dim is not specified, reduce the last dimension.
            int dim_ = (dim is null)
                ? tensor.Shape.Count - 1 // [^1] by default
                : (dim.Value.IsFromEnd)
                    ? tensor.Shape.Count - dim.Value.Value
                    : dim.Value.Value;

            // Compute the tensor if it is not already computed.
            // Remember that a tensor without operations is already computed.
            if (tensor.OperandCount != 0)
                tensor = Compute<T, TOp>(tensor, dim_);

            // If the dimension to reduce is already 1, return the tensor.
            if (tensor.Shape[dim_] == 1)
                return (TensorData<T>)tensor;

            // Compute the shape of the result tensor.
            var sourceShape = tensor.Shape;
            int resultingSize = (tensor.Shape[dim_] + ReduceKernelElementsCount - 1) / ReduceKernelElementsCount;
            var destinationShape = tensor.Shape.SetDim(dim_, resultingSize);

            var shapeGpu = MMU.GetBuffer((int[])sourceShape);
            var destinationGpu = MMU.GetBuffer<T>(destinationShape.Length);

            var fnc = Accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<T, Stride1D.Dense>,
                int,
                int,
                SpecializedValue<int>>(ReduceKernel<T, TOp>);

            if (tensor is TensorData<T> tensorData)
            {
                fnc(
                    new Index1D((int)destinationShape.Length),
                    tensorData.View,
                    MMU.GetBuffer((int[])tensorData.Shape).AcceleratorData.View,
                    destinationGpu.AcceleratorData.View,
                    dim_,
                    ReduceKernelElementsCount,
                    new SpecializedValue<int>(tensorData.Shape.Count));

                // Reduce the tensor until it has only one element in the dimension to reduce.
                AcceleratorBuffer<T>? sourceGpu = null;
                while (resultingSize > 1)
                {
                    (sourceGpu, destinationGpu) = (destinationGpu, sourceGpu);

                    sourceShape = destinationShape;

                    resultingSize = (resultingSize + ReduceKernelElementsCount - 1) / ReduceKernelElementsCount;
                    destinationShape = sourceShape.SetDim(dim_, resultingSize);

                    destinationGpu ??= MMU.GetBuffer<T>(destinationShape.Length); // Allocate the destination buffer the first time.

                    fnc(
                        new Index1D((int)destinationShape.Length),
                        sourceGpu.AcceleratorData.View,
                        MMU.GetBuffer((int[])sourceShape).AcceleratorData.View,
                        destinationGpu.AcceleratorData.View,
                        dim_,
                        ReduceKernelElementsCount,
                        new SpecializedValue<int>(sourceShape.Count));
                }
                if (sourceGpu is not null)
                {
                    sourceGpu.Dispose();
                    sourceGpu = destinationGpu;
                    destinationGpu = MMU.GetBuffer<T>(destinationShape.Length);
                    destinationGpu.AcceleratorData.CopyFrom(sourceGpu);
                    sourceGpu.Dispose();
                }
                return new TensorData<T>($"{tensor.Name} {{result}}", new Shape(destinationShape), destinationGpu);
            }
            else
                throw new Exception($"This should not happen.");
        }
    }
}
