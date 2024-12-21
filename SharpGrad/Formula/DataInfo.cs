using ILGPU.Runtime;
using SharpGrad.Formula.Internal;
using System;

namespace SharpGrad.Formula
{
    public readonly struct DataInfo<TResult>(BIndex<ushort> gradientIndex)
    {
        public DataInfo(int gradientIndex)
            : this(gradientIndex < 0
                  ? BIndex<ushort>.Empty
                  : gradientIndex < ushort.MaxValue
                    ? (BIndex<ushort>)gradientIndex 
                  : throw new ArgumentOutOfRangeException(nameof(gradientIndex)))
        { }

        public readonly BIndex<ushort> GradientIndex = gradientIndex;

        public readonly bool IsGradiable => !GradientIndex.IsEmpty;

        internal void SetAccelerator(Accelerator accelerator)
        {
            throw new NotImplementedException();
        }
    }
}