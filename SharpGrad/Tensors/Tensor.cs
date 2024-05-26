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
    public partial class Tensor<TType>(Shape shape) : TensorBase<TType>(shape), ITensor<Tensor<TType>, TType>,
        IAdditionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        ISubtractionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IMultiplyOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IDivisionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        private readonly DeviceBuffer<TType> data = new(shape.Size);
        protected override DeviceBuffer<TType> Data { get => data; }
        private readonly DeviceBuffer<TType> gradients = new(shape.Size);

        public override TType this[params int[] indices]
        {
            get => Data.CPUData[shape.GetFlattenedIndex(indices)];
            set => Data.CPUData[shape.GetFlattenedIndex(indices)] = value;
        }


        public Tensor(Shape shape, TType[] data) : this(shape)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException($"Expected data length {shape.Aggregate(1, (a, b) => a * b)}, got {data.Length}");
        }

        public Tensor(params Dim[] shape) : this(new Shape(shape)) { }

        public void AddGradient(Tensor<TType> gradient)
        {
            if(gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");

            if (gradients != null)
                ExecGpu(AddOp<TType>.Apply, gradients, gradient.Data, gradients);
        }

        public static Tensor<TType> operator +(Tensor<TType> left, Tensor<TType> right) => ExecGpu(AddOp<TType>.Apply, left, right);
        public static Tensor<TType> operator -(Tensor<TType> left, Tensor<TType> right) => ExecGpu(SubOp<TType>.Apply, left, right);
        public static Tensor<TType> operator *(Tensor<TType> left, Tensor<TType> right) => ExecGpu(MulOp<TType>.Apply, left, right);
        public static Tensor<TType> operator /(Tensor<TType> left, Tensor<TType> right) => ExecGpu(DivOp<TType>.Apply, left, right);
    }
}
