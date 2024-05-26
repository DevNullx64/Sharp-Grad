using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    internal class TensorOperationOne<TType>(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> fnc, Tensor<TType> left) : ITensor<TensorOperationOne<TType>, TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public readonly Tensor<TType> LeftOperand = left;

        public readonly Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> Fnc = fnc;

        public Shape Shape => throw new NotImplementedException();

        public TType this[params int[] indices] {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException(); }

        public static TensorOperationOne<TType> operator +(TensorOperationOne<TType> left, TensorOperationOne<TType> right) => new(AddOp<TType>.Apply, left.LeftOperand);
        public static TensorOperationOne<TType> operator -(TensorOperationOne<TType> left, TensorOperationOne<TType> right) => new(SubOp<TType>.Apply, left.LeftOperand);
        public static TensorOperationOne<TType> operator *(TensorOperationOne<TType> left, TensorOperationOne<TType> right) => new(MulOp<TType>.Apply, left.LeftOperand);
        public static TensorOperationOne<TType> operator /(TensorOperationOne<TType> left, TensorOperationOne<TType> right) => new(DivOp<TType>.Apply, left.LeftOperand);
    }
}
