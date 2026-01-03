using System;

namespace SharpGrad.DifEngine
{
    /// <summary>
    /// Base class for all values in the computation graph.
    /// </summary>
    public abstract class ValueBase
    {
        public readonly string Name;
        /// <summary>
        /// The numeric type of the value.
        /// </summary>
        public readonly Type Type;

        /// <summary>
        /// The shape of the value.
        /// </summary>
        public Shape Shape { get; private set; }

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

        public ValueBase[] Operands;

        public ValueBase(string name, Type type, Shape shape, params ValueBase[] childs)
        {
            Name = name;
            Type = type;
            Shape = shape;
            Operands = childs;
        }
    }
}