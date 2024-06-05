using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class TensorData<T>(Shape shape) : Tensor<T>(shape)
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        internal readonly AcceleratorBuffer<T> data = Acc.GetAcceleratorBuffer<T>(shape.Length);
        internal override ArrayView1D<T, Stride1D.Dense> GetArrayView1D() => data.AcceleratorData.View;

        public override T this[params Index[] indices] 
        {
            get => data[Shape.GetFlattenIndex(indices)];
            set => data[Shape.GetFlattenIndex(indices)] = value;
        }

        public override T[,] this[params Range[] ranges] 
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public TensorData(params Dim[] dims) : this(new Shape(dims)) { }

    }
}
