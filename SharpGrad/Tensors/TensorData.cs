using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{

    public class TensorData<T> : TensorConst<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        private AcceleratorBuffer<T>? gradientBuffer;
        internal AcceleratorBuffer<T> GradientBuffer => gradientBuffer ?? throw new InvalidOperationException("Gradient is disabled for this tensor.");

        /// <summary>
        /// Whether the tensor needs gradient.
        /// </summary>
        public override bool NeedsGradient => gradientBuffer is not null;

        /// <summary>
        /// Disable gradient computation for this tensor.
        /// </summary>
        public void DisableGradient() => gradientBuffer = null;

        /// <summary>
        /// Enable gradient computation for this tensor.
        /// </summary>
        public void EnableGradient() => gradientBuffer = KernelProcessUnit.DefaultKPU.MMU.GetBuffer<T>(Shape.Length);

        public new T this[params Index[] indices]
        {
            get => base[indices];
            set
            {
                int flattenedIndex = Shape.GetFlattenIndex(indices);
                Buffer[flattenedIndex] = value;
            }
        }

        internal TensorData(string name, Shape shape, AcceleratorBuffer<T> buffer, bool needGradient = true)
            : base(name, shape, buffer)
        { if (needGradient) EnableGradient(); }

        protected TensorData(Shape shape, AcceleratorBuffer<T> buffer, bool needGradient = true)
            : this(GetNextName(), shape, buffer, needGradient)
        { }


        public TensorData(string name, Shape shape, bool needGradient = true)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer<T>(shape.Length), needGradient)
        { }

        public TensorData(Shape shape, bool needGradient = true)
            : this(GetNextName(), shape, needGradient)
        { }

        public TensorData(string name, Shape shape, T[] data, bool needGradient = true)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer(data), needGradient)
        { }

        public TensorData(Shape shape, T[] data, bool needGradient = true)
            : this(GetNextName(), shape, data, needGradient)
        { }

        public static implicit operator TensorData<T>((string Name, Shape Shape) tensor)
            => new(tensor.Name, tensor.Shape);
        public static implicit operator TensorData<T>((Shape Shape, string Name) tensor)
            => new(tensor.Name, tensor.Shape);

        public static implicit operator TensorData<T>((string Name, Shape Shape, T[] Data) tensor)
            => new(tensor.Name, tensor.Shape, tensor.Data);
        public static implicit operator TensorData<T>((Shape Shape, string Name, T[] Data) tensor)
            => new(tensor.Name, tensor.Shape, tensor.Data);

        public override int GetHashCode() => HashCode.Combine(Name, Shape);
    }
}
