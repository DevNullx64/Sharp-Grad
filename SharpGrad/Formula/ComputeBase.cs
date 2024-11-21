using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula
{
    /// <summary>
    /// Represents a computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal abstract class ComputeBase<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        internal ComputeBase(Result<TResult> result, OpCode opCode, params int[] operands)
            : base(result, opCode, operands)
        {
            foreach (var operand in operands)
                Get(operand).ResetCompute += OnResetComputeHandler;
            Add(this);
        }

        private void OnResetComputeHandler()
        {
            if (IsOuput)
                IsComputed = false;
            else
                OnResetCompute();
        }
    }
}