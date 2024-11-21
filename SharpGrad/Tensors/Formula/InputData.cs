using SharpGrad.Tensors.KPU;
using System;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors.Formula
{
    /// <summary>
    /// Represents an input data element.
    /// </summary>
    /// <typeparam name="TResult">The type of the input data element.</typeparam>
    public class InputData<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public TResult[] Value
        {
            get => Result.Content!.SafeCPUData;
            set
            {
                if (!Value.SequenceEqual(value))
                {
                    Result.Content!.SafeCPUData = value;
                    OnResetCompute();
                }
            }
        }

        internal InputData(Result<TResult> value, bool isGradiable = true)
            : base(value, OpCode.Store)
        {
            IsGradiable = isGradiable;
            Add(this);
        }

        public InputData(Shape shape, TResult[]? data, bool isGradiable)
            : this(new Result<TResult>(shape, isGradiable, data is not null, false))
        {
            if (data is not null)
                Value = data;
        }

        protected override bool OperandsEquals(params int[] operands)
            => Result.Equals(Get(operands[0]).Result);

        protected override int GetOperandsHashCode()
            => Result.GetHashCode();
    }
}