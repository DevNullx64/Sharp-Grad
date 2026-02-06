using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public class CastedValue<TFrom, TTo>(Value<TFrom> input) :
        ComputedMixedValue<TTo>(KindGraphNode.Cast.GetResultShape(input.Shape), KindGraphNode.Cast, input)
        where TFrom : struct, INumber<TFrom>
        where TTo : struct, INumber<TTo>
    {
        public Value<TFrom> InputOperand
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => input;
        }
        public Value<TTo> OutputOperand
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => this;
        }
    }
}