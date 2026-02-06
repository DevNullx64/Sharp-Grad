using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad
{
    public class Variable<TType> : Value<TType>
        where TType : struct, INumber<TType>
    {
        public new TType this[Dimdices indices] {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => base[indices];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => base[indices] = value;
        }

        public Variable(string name, Array data, Shape shape)
            : base(shape, name, KindGraphNode.Variable)
        {
            if(shape.Rank != data.Rank)
            {
                throw new ArgumentException($"The shape rank {shape.Rank} is not equal to the data rank {data.Rank}");
            }
            for (int d = 0; d < data.Rank; d++)
            {
                if (shape[d].Size != data.GetLength(d))
                {
                    throw new ArgumentException($"The shape dimension size {shape[d].Size} is not equal to the data dimension length {data.GetLength(d)} at dimension {d}");
                }
            }
            base.data.SetData(data);
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
            this(name, new TType[shape.Size], shape)
        { }
    }
}