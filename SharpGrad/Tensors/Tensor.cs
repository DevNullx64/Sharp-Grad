using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    public partial class Tensor<T>(Shape shape) : TensorBase<T>(shape), ITensor<Tensor<T>, T>,
        IAdditionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IMultiplyOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IDivisionOperators<Tensor<T>, Tensor<T>, Tensor<T>>
        where T : unmanaged, IFloatingPoint<T>
    {
        private readonly DeviceBuffer<T> data = new(shape.Size);
        internal override DeviceBuffer<T> Data { get => data; }
        private readonly DeviceBuffer<T> gradients = new(shape.Size);

        public override T this[params int[] indices]
        {
            get => Data.CPUData[shape.GetFlattenedIndex(indices)];
            set => Data.CPUData[shape.GetFlattenedIndex(indices)] = value;
        }


        public Tensor(Shape shape, T[] data) : this(shape)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException($"Expected data length {shape.Aggregate(1, (a, b) => a * b)}, got {data.Length}");
        }

        public Tensor(params Dim[] shape) : this(new Shape(shape)) { }

        public void AddGradient(Tensor<T> gradient)
        {
            if(gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");

            if (gradients != null)
                ExecGpu(AddOp<T>.Apply, gradients, gradient.Data, gradients);
        }

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => ExecTensorOnGpu(AddOp<T>.Apply, left, right);
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => ExecTensorOnGpu(SubOp<T>.Apply, left, right);
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => ExecTensorOnGpu(MulOp<T>.Apply, left, right);
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => ExecTensorOnGpu(DivOp<T>.Apply, left, right);
    }
}
