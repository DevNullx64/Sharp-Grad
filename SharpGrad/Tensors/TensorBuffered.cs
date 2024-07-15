using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class TensorBuffered<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal readonly AcceleratorBuffer<T> buffer;
        internal ArrayView1D<T, Stride1D.Dense> View => buffer.AcceleratorData.View;
        public override long Depth => 0;
        public override int OperandCount => 0;

        public override T this[params Index[] indices]
        {
            get
            {
                var flattenedIndex = Shape.GetFlattenIndex(indices);
                return buffer[flattenedIndex];
            }
        }

        protected TensorBuffered(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape)
        {
            Shape = shape;
            this.buffer = buffer;
        }
    }
}
