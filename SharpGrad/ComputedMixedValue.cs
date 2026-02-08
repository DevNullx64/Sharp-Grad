using SharpGrad.DifEngine.SyntaxBuilder;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    /// <summary>
    /// Base class for values computed by applying an operation.
    /// Supports mixed-type operands (e.g., for Cast operations).
    /// </summary>
    public abstract class ComputedMixedValue<T> :
        Value<T>
        where T : struct, INumber<T>
    {
        public ComputedMixedValue(Shape shape, KindGraphNode kind, params Value[] operands)
            : base(shape, kind.ToSymbol(), kind)
        {
            if (kind.IsValue())
            {
                throw new ArgumentException("Computed values cannot be of kind Variable or Constant.", nameof(kind));
            }
        }

        public bool IsMixed
        {
            get
            {
                if (Kind.IsBinary())
                {
                    IGraphNodeBinary<Value> binaryNode = (IGraphNodeBinary<Value>)this;
                    return binaryNode.Left.Data.ElementType != binaryNode.Right.Data.ElementType ||
                           binaryNode.Left.Data.ElementType != typeof(T);
                }
                else if (Kind.IsUnary())
                {
                    IGraphNodeUnary<Value> unaryNode = (IGraphNodeUnary<Value>)this;
                    return unaryNode.Operand.Data.ElementType != typeof(T);
                }
                else
                {
                    return false;
                }
            }
        }
    }
}   