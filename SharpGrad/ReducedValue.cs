using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public class ReducedValue<T>(KindReduction kind, Value<T> Input, params Dimension[] reduceDims) :
        ComputedMixedValue<T>(kind.GetResultShape(Input.Shape, reduceDims), (KindGraphNode)kind, Input)
        where T : struct, INumber<T>
    {
        public Dimension[] ReduceDims
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => reduceDims;
        }
        public Value<T> InputOperand
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Input;
        }
    }
}