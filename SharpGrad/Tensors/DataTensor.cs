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
    public partial class DataTensor<T, TGrad>(Shape shape, bool isGrad = false) : Tensor<T, TGrad>(shape, isGrad), ITensor<DataTensor<T, TGrad>, T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        private readonly AcceleratorBuffer<T> data = new(shape.Size);
        internal override AcceleratorBuffer<T> Data { get => data; }

        public override T this[params int[] indices]
        {
            get => Data.CPUData[shape.GetFlattenedIndex(indices)];
            set => Data.CPUData[shape.GetFlattenedIndex(indices)] = value;
        }


        public DataTensor(Shape shape, T[] data) : this(shape)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException($"Expected data length {shape.Aggregate(1, (a, b) => a * b)}, got {data.Length}");
        }

        public DataTensor(params Dim[] shape) : this(new Shape(shape)) { }

        public static Tensor<T, TGrad> operator +(DataTensor<T, TGrad> left, DataTensor<T, TGrad> right) => new TensorOperationTwo<T, AddOp<T, TGrad>, TGrad>(left, right);

        public static Tensor<T, TGrad> operator -(DataTensor<T, TGrad> left, DataTensor<T, TGrad> right) => new TensorOperationTwo<T, SubOp<T, TGrad>, TGrad>(left, right);

        public static Tensor<T, TGrad> operator *(DataTensor<T, TGrad> left, DataTensor<T, TGrad> right) => new TensorOperationTwo<T, MulOp<T, TGrad>, TGrad>(left, right);

        public static Tensor<T, TGrad> operator /(DataTensor<T, TGrad> left, DataTensor<T, TGrad> right) => new TensorOperationTwo<T, DivOp<T, TGrad>, TGrad>(left, right);
    }

}
