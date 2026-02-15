using SharpGrad.DifEngine.SyntaxBuilder;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public class Variable<TType> : Value<TType>
        where TType : struct, INumber<TType>
    {
        public Variable(string name, Shape shape)
            : base(shape, name, KindGraphNode.Variable)
        { }

        public Variable(string name, Array data, Shape shape)
            : this(shape, name)
        {
            base.data.SetData(data);
        }
        public new DataBuffer<TType> Data
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => data;
        }

        public Variable(string name, TType data)
            : this(name, new TType[] { data }, [])
        { }
        public Variable(string name, TType[] data, Dimension d1)
            : this(name, data, new Shape(d1))
        { }
        public Variable(string name, TType[,] data, Dimension d1, Dimension d2)
            : this(name, data, new Shape(d1, d2))
        { }
        public Variable(string name, TType[,,] data, Dimension d1, Dimension d2, Dimension d3)
            : this(name, data, new Shape(d1, d2, d3))
        { }
        public Variable(string name, TType[,,,] data, Dimension d1, Dimension d2, Dimension d3, Dimension d4)
            : this(name, data, new Shape(d1, d2, d3, d4))
        { }
        public Variable(string name, TType[,,,,] data, Dimension d1, Dimension d2, Dimension d3, Dimension d4, Dimension d5)
            : this(name, data, new Shape(d1, d2, d3, d4, d5))
        { }
        public Variable(string name, TType[,,,,,] data, Dimension d1, Dimension d2, Dimension d3, Dimension d4, Dimension d5, Dimension d6)
            : this(name, data, new Shape(d1, d2, d3, d4, d5, d6))
        { }
        public Variable(string name, TType[,,,,,,] data, Dimension d1, Dimension d2, Dimension d3, Dimension d4, Dimension d5, Dimension d6, Dimension d7)
            : this(name, data, new Shape(d1, d2, d3, d4, d5, d6, d7))
        { }

        public Variable(Shape shape, string name) :
            this(name, shape)
        { }
    }
}