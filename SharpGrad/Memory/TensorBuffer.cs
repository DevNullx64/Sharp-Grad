using ILGPU.Runtime;
using SharpGrad.Tensors;
using System;
using System.Numerics;

namespace SharpGrad.Memory
{
    public readonly struct TensorBuffer<T> where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal readonly AcceleratorBuffer<T> AcceleratorBuffer;
        public readonly Shape Shape;

        public Accelerator Device => KernelProcessUnit.DefaultKPU.MMU.Accelerator;

        internal TensorBuffer(AcceleratorBuffer<T> buffer, Shape shape)
        {
            AcceleratorBuffer = buffer;
            Shape = shape;
        }

        public TensorBuffer<T> Reshape(Shape newShape)
        {
            if (Shape.Length != newShape.Length)
                throw new ArgumentException($"Cannot reshape a tensor of length {Shape.Length} to a tensor of length {newShape.Length}.");
            return new(AcceleratorBuffer, newShape);
        }
    }
}