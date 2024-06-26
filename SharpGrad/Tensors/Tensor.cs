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

        public int OperandCound => 0;

        public abstract T this[params Index[] indices] { get; }

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, AddOp<T>>(left, right);
        public static Tensor<T> operator -(Tensor<T> value) => new TensorOperation1<T, NegOp<T>>(value);
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, SubOp<T>>(left, right);
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, MulOp<T>>(left, right);
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => new TensorOperation2<T, DivOp<T>>(left, right);


        public static implicit operator Tensor<T>(T[] data) => new TensorData<T>(GetNextName(), new Shape(data.Length), data);
        public static implicit operator Tensor<T>((string Name, T[] Data) tensor) => new TensorData<T>(tensor.Name, new Shape(tensor.Data.Length), tensor.Data);
        public static implicit operator Tensor<T>((T[] Data, string Name) tensor) => new TensorData<T>(tensor.Name, new Shape(tensor.Data.Length), tensor.Data);

        public abstract bool Equals(ITensor? other);

        internal virtual void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (!topoSort.TryGetValue(this, out DfsNode<T>? node))
            {
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public Dictionary<Tensor<T>, DfsNode<T>> DepthFirstSearch()
        {
            Dictionary<Tensor<T>, DfsNode<T>> topoSort = new();
            DepthFirstSearch(topoSort);
            return topoSort;
        }
        public abstract void Backward();

        public static bool operator ==(Tensor<T>? left, Tensor<T>? right) => left is null ? right is null : left.Equals(right);
        public static bool operator !=(Tensor<T>? left, Tensor<T>? right) => left is null ? right is not null : !left.Equals(right);
    }

    internal abstract class Tensor<T, TOp>(Shape shape) : Tensor<T>(TOp.Symbol, shape), ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecutor
    {
        public OpCode OpCode => TOp.OpCode;

        public abstract int OperandCound { get; }
    }
}
