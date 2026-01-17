using SharpGrad.ExprLambda;
using System;
using System.Numerics;

namespace SharpGrad.DifEngine
{
    /// <summary>
    /// Base class for all values in the computation graph.
    /// </summary>
    public abstract class Value : IGraphNode<Value>
    {
        public readonly string Name;
        /// <summary>
        /// The numeric type of the value.
        /// </summary>
        public Type Type { get; }

        /// <summary>
        /// The shape of the value.
        /// </summary>
        public Shape Shape { get; }

        /// <summary>
        /// The size of the value.
        /// </summary>
        public int Size => Shape.Size;

        /// <summary>
        /// Whether the value is a scalar.
        /// </summary>
        public bool IsScalar => Shape.IsScalar;

        /// <summary>
        /// If true, this value is an output of the computation graph and should be saved back to its data field.
        /// </summary>
        public bool IsOutput { get; set; } = false;

        public Value[] Operands { get; }

        public bool IsParallelBarrier { get; }

        public Value(string name, Type type, Shape shape, bool isParallelBarrier, params Value[] childs)
        {
            Name = name;
            Type = type;
            Shape = shape;
            IsParallelBarrier = isParallelBarrier;
            Operands = childs;
        }
        public override string ToString() => Name;
    }
}