using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{

    public class TensorData<T> : TensorConst<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        private bool needsGradient = true;
        public override bool NeedsGradient => needsGradient;

        internal readonly AcceleratorBuffer<T> GradientBuffer;
        public void DisableGradient() => needsGradient = false;
        public void EnableGradient() => needsGradient = true;

        public new T this[params Index[] indices]
        {
            get => base[indices];
            set
            {
                int flattenedIndex = Shape.GetFlattenIndex(indices);
                buffer[flattenedIndex] = value;
            }
        }

        internal TensorData(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape, buffer)
        { GradientBuffer = KernelProcessUnit.DefaultKPU.MMU.GetBuffer<T>(shape.Length); }

        protected TensorData(Shape shape, AcceleratorBuffer<T> buffer)
            : this(GetNextName(), shape, buffer)
        { }


        public TensorData(string name, Shape shape)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer<T>(shape.Length))
        { }

        public TensorData(Shape shape)
            : this(GetNextName(), shape)
        { }

        public TensorData(string name, Shape shape, T[] data)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer(data))
        { }

        public TensorData(Shape shape, T[] data)
            : this(GetNextName(), shape, data)
        { }

        public static implicit operator TensorData<T>((string Name, Shape Shape) tensor)
            => new(tensor.Name, tensor.Shape);
        public static implicit operator TensorData<T>((string Name, Shape Shape, T[] Data) tensor)
            => new(tensor.Name, tensor.Shape, tensor.Data);

        public static implicit operator TensorData<T>((Shape Shape, string Name) tensor)
            => new(tensor.Name, tensor.Shape);
        public static implicit operator TensorData<T>((Shape Shape, string Name, T[] Data) tensor)
            => new(tensor.Name, tensor.Shape, tensor.Data);

        public override int GetHashCode() => HashCode.Combine(buffer.CPUData);
    }
}
