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

        /// <summary>
        /// Gets the typed DataBuffer of type T from an untyped DataBuffer.
        /// </summary>
        /// <typeparam name="T">The type of the DataBuffer.</typeparam>
        /// <param name="buffer">The untyped DataBuffer.</param>
        /// <returns>The typed <see cref="DataBuffer{T}"/>.</returns>
        /// <remarks>
        /// Throws an InvalidOperationException if the buffer is not of the expected type.
        /// </remarks>
        private static DataBuffer<T> GetBuffer<T>(DataBuffer? buffer)
            where T : struct, INumber<T>
        {
            if (buffer is not DataBuffer<T> typeBuffer)
            {
                throw new InvalidOperationException($"Expected buffer of type {typeof(DataBuffer<T>).Name}, but got {buffer?.GetType().Name ?? "null"}.");
            }
            return typeBuffer;
        }
    }
}
