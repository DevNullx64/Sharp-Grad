using ILGPU.Runtime;
using SharpGrad.Formula.Internal;
using System;

namespace SharpGrad.Formula
{
    public readonly struct DataInfo<TResult>(BIndex<ushort> gradientIndex)
    {
        public readonly BIndex<ushort> GradientIndex = gradientIndex;

        public readonly bool IsGradiable => !GradientIndex.IsEmpty;

        internal void SetAccelerator(Accelerator accelerator)
        {
            throw new NotImplementedException();
        }
    }
}