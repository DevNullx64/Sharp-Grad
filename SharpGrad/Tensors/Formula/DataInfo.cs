using ILGPU.Runtime;
using System;

namespace SharpGrad.Tensors.Formula
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