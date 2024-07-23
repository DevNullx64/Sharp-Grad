using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class Tensor<T>(string name, Shape shape) : ITensor<T>,
        IAdditionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IUnaryNegationOperators<Tensor<T>, Tensor<T>>,
        IMultiplyOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IDivisionOperators<Tensor<T>, Tensor<T>, Tensor<T>>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        private static object lockObj = new();
        private static int nextId = 0;
        protected static string GetNextName()
        {
            lock (lockObj)
                return $"T{nextId++}";
        }

        public string Name { get; } = name;

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
        public abstract bool NeedsGradient { get; }

        public abstract int OperandCount { get; }

        public abstract T this[params Index[] indices] { get; }

        public Tensor<T> Sum(Index? dim = null) => KernelProcessUnit.DefaultKPU.Reduce<T, AddOp<T>>(this, dim);
        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, AddOp<T>>(left, right);
        public static Tensor<T> operator -(Tensor<T> value) => new TensorOperation1<T, NegOp<T>>(value);
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, SubOp<T>>(left, right);
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, MulOp<T>>(left, right);
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, DivOp<T>>(left, right);


        public static implicit operator Tensor<T>(T[] data) => new TensorConst<T>(GetNextName(), new Shape(data.Length), data);
        public static implicit operator Tensor<T>((string Name, T[] Data) tensor) => new TensorData<T>(tensor.Name, new Shape(tensor.Data.Length), tensor.Data);
        public static implicit operator Tensor<T>((T[] Data, string Name) tensor) => new TensorData<T>(tensor.Name, new Shape(tensor.Data.Length), tensor.Data);

        public abstract bool Equals(ITensor? other);

        internal virtual void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (topoSort.TryGetValue(this, out DfsNode<T>? node))
                node.UsageCount++;
            else
            {
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public Dictionary<Tensor<T>, DfsNode<T>> DepthFirstSearch()
        {
            Dictionary<Tensor<T>, DfsNode<T>> topoSort = [];
            DepthFirstSearch(topoSort);
            return topoSort;
        }
        public abstract void Backward();

        public static bool operator ==(Tensor<T>? left, Tensor<T>? right) => left is null ? right is null : left.Equals(right);
        public static bool operator !=(Tensor<T>? left, Tensor<T>? right) => left is null ? right is not null : !left.Equals(right);
    }
}
