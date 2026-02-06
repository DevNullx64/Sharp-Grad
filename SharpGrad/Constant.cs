using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Numerics;

namespace SharpGrad
{
    public class Constant<TType> : Value<TType>
        where TType : struct, INumber<TType>
    {
        private static int InstanceCount = 0;

        public Constant(TType[] data, Shape shape, string name)
            : base(shape, name, KindGraphNode.Constant)
        {
            if (shape.Size != data.Length)
            {
                throw new ArgumentException($"The shape size {shape.Size} is not equal to the data length {data.Length}");
            }
            base.data.SetData(data);
            IsOutput = false;
        }

        public Constant(TType data, string name)
            : this([data], [], name)
        { }

        public static implicit operator Constant<TType>(TType d)
            => new(d, $"c{InstanceCount++}");

        public override string ToString()
            => Shape.ToString();
    }
}