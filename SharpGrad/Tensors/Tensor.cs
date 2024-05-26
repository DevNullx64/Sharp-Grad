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
    public partial class Tensor<T, TGrad>(Shape shape) : TensorBase<T, TGrad>(shape), ITensor<Tensor<T, TGrad>, T>,
        IAdditionOperators<Tensor<T, TGrad>, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        ISubtractionOperators<Tensor<T, TGrad>, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        IMultiplyOperators<Tensor<T, TGrad>, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        IDivisionOperators<Tensor<T, TGrad>, Tensor<T, TGrad>, Tensor<T, TGrad>>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        private readonly AcceleratorBuffer<T> data = new(shape.Size);
        internal override AcceleratorBuffer<T> Data { get => data; }

        private readonly AcceleratorBuffer<T> gradients = new(shape.Size);

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

        public void AddGradient(Tensor<T, TGrad> gradient)
        {
            if(gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");

            if (gradients != null)
                ExecGpu(AddOp<T, TGrad>.ApplyGpu, gradients, gradient.Data, gradients);
        }

        public static Tensor<T, TGrad> operator +(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => ExecTensorOnGpu(AddOp<T, TGrad>.ApplyGpu, left, right);
        public static Tensor<T, TGrad> operator -(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => ExecTensorOnGpu(SubOp<T, TGrad>.ApplyGpu, left, right);
        public static Tensor<T, TGrad> operator *(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => ExecTensorOnGpu(MulOp<T, TGrad>.ApplyGpu, left, right);
        public static Tensor<T, TGrad> operator /(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => ExecTensorOnGpu(DivOp<T, TGrad>.ApplyGpu, left, right);
    }
}
