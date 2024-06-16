using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad.Tensors
{
    public abstract class Tensor<T>(Shape shape) : ITensor<T>,
        IAdditionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IUnaryNegationOperators<Tensor<T>, Tensor<T>>,
        IMultiplyOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IDivisionOperators<Tensor<T>, Tensor<T>, Tensor<T>>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        protected Shape Shape_ = shape;
        public virtual Shape Shape
        {
            get => Shape_;
            set
            {
                if (Shape_.Length != value.Length)
                    throw new InvalidOperationException($"Cannot update shapes {Shape_} and {value}");
                Shape_ = value;
            }
        }
        public long Length => Shape.Length;
        public abstract long Depth { get; }
        public abstract T this[params Index[] indices] { get; }

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, AddOp<T>>(left, right);
        public static Tensor<T> operator -(Tensor<T> value) => new StreamTensor1<T, NegOp<T>>(value);
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, SubOp<T>>(left, right);
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, MulOp<T>>(left, right);
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, DivOp<T>>(left, right);


        public static implicit operator Tensor<T>(T[] data) => new DataTensor<T>(new Shape(data.Length), data);

        public abstract bool Equals(ITensor? other);
    }

    internal abstract class Tensor<T, TOp>(Shape shape) : Tensor<T>(shape), ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor
    {
        public OpCode OpCode => TOp.OpCode;

        public abstract void DepthFirstSearch(List<ITensorOperation<T>> topoSort, int level, Dictionary<Tensor<T>, (int UsageCount, int Level)> visited, Dictionary<Tensor<T>, int> leaf);
        public abstract void Backward();
    }
}
