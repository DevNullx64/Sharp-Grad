using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        ParallelOptions _parallelOptions;
        public int ThreadCount
        {
            get => _parallelOptions.MaxDegreeOfParallelism;
            set => _parallelOptions.MaxDegreeOfParallelism = value;
        }

        public DeviceCpu(int cpuCount)
        {
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = cpuCount };
        }
        public DeviceCpu() :
            this(Environment.ProcessorCount)
        { }


        public bool IsAvailable() => true;

        public string Name => nameof(DeviceCpu);

        public void ResetGradient<TType>(Value<TType> loss)
            where TType : struct, INumber<TType>
        {
            loss.untypedGrad = null;
        }
    }
}
