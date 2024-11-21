using SharpGrad.Tensors.KPU;
using SharpGrad.Tensors.Operators;
using System;
using System.Numerics;

namespace SharpGrad.Tensors.Formula
{
    /// <summary>
    /// Represents a binary computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal class ComputeBinaryClass<TOp, TResult>(int leftIndex, int rightIndex, bool isGradiable) :
        ComputeBase<TResult>(new(TOp.ResultingShape(Get(leftIndex).Shape, Get(rightIndex).Shape), isGradiable, false, true), TOp.OpCode, leftIndex, rightIndex)
        where TOp : IExecBinary<TResult, TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Left => Get(OperandIndices[0]);
        public ComputeElement<TResult> Right => Get(OperandIndices[1]);

        protected override bool OperandsEquals(params int[] operandIndeces)
        {
            bool result = OperandIndices[0] == operandIndeces[0] && OperandIndices[1] == operandIndeces[1];
            if (!result && (OpCode & OpCode.IsCommutative) == OpCode.IsCommutative)
                result = OperandIndices[0] == operandIndeces[1] && OperandIndices[1] == operandIndeces[0];
            return result;
        }

        protected override int GetOperandsHashCode()
        {
            int hash = Operands[0].GetHashCode();
            if ((OpCode & OpCode.IsCommutative) == OpCode.IsCommutative)
                for (int i = 1; i < OperandsLength; i++)
                    hash ^= Operands[i].GetHashCode();
            else
                for (int i = 1; i < OperandsLength; i++)
                    hash = hash * 31 + Operands[i].GetHashCode();
            return HashCode.Combine(OpCode, hash);
        }
    }
}