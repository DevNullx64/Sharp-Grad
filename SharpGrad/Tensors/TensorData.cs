using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{

    public class TensorData<T> : TensorBuffered<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {

        public new T this[params Index[] indices]
        {
            get => base[indices];
            set
            {
                int flattenedIndex = Shape.GetFlattenIndex(indices);
                buffer[flattenedIndex] = value;
            }
        }

        public void Set(T value, params Index[] indices)
        {
            var flattenedIndex = Shape.GetFlattenIndex(indices);
            buffer[flattenedIndex] = value;
        }

        internal TensorData(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape, buffer)
        { }

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

        public override bool Equals(ITensor? other)
            => other is TensorData<T> tensor && buffer == tensor.buffer;

        public override bool Equals(object? obj) => obj is TensorData<T> tensor && Equals(tensor);

        public override string ToString() => $"{Name}{Shape}";

        public override void Backward()
        {
            throw new NotSupportedException($"TensorData<{typeof(T).Name}> does not support backward operation");
        }

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
