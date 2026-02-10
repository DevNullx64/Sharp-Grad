using SharpGrad.DifEngine.SyntaxBuilder;
using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Numerics;

namespace SharpGrad
{
    public class UnaryComputedValue<T>(KindUnary kind, Value<T> input) :
        ComputedMixedValue<T>(kind.GetResultShape(input.Shape), (KindGraphNode)kind, input),
        IGraphNodeUnary<Value>
        where T : struct, INumber<T>
    {
        public Value<T> Input
        {
            get => input;
        }

        public new KindUnary Kind
        {
            get => (KindUnary)base.Kind;
        }

        public Value Operand
        {
            get => input;
        }
    }
}