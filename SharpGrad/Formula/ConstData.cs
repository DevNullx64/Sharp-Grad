using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula
{
    public class ConstData<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public TResult[] Value => Result.Content!.SafeCPUData;

        internal ConstData(Result<TResult> value)
            : base(value, OpCode.Store)
        {
            Add(this);
        }

        protected override bool OperandsEquals(params int[] operandIndeces)
            => Result.Equals(Get(operandIndeces[0]).Result);

        protected override int GetOperandsHashCode()
            => Result.GetHashCode();
    }
}