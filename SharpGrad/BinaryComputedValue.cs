using SharpGrad.DifEngine.SyntaxBuilder;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public class BinaryComputedValue<T>(KindBinary kind, Value<T> left, Value<T> right) :
        ComputedMixedValue<T>(kind.GetResultShape(left.Shape, right.Shape), (KindGraphNode)kind),
        IGraphNodeBinary<Value>
        where T : struct, INumber<T>
    {
        public Value Left
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => left;
        }

        public Value Right
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => right;
        }

        public new KindBinary Kind
        {
            get => (KindBinary)base.Kind;
        }
    }
}