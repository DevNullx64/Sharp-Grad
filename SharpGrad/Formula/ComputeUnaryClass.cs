using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula
{
    /// <summary>
    /// Represents a unary computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal class ComputeUnaryClass<TOp, TResult>(int operandIndex, bool isGradiable) :
        ComputeBase<TResult>(new(TOp.ResultingShape(Get(operandIndex).Shape), isGradiable, false, true), TOp.OpCode, operandIndex)
        where TOp : IExecUnary<TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Operand
            => Operands[0];

        protected override bool OperandsEquals(params int[] operandIndeces)
            => OperandIndices[0] == operandIndeces[0];

        protected override int GetOperandsHashCode()
            => Operand.GetHashCode();
    }
}