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
        /// Throws an InvalidOperationException if the DataBuffer is not initialized.
        /// </summary>
        /// <typeparam name="T">The type of the DataBuffer.</typeparam>
        /// <param name="data">The DataBuffer to check.</param>
        /// <exception cref="InvalidOperationException">Thrown if the DataBuffer is not initialized.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ThrowIfNotInitialized<T>(DataBuffer<T> data)
            where T : struct, INumber<T>
        {
            if (!data.IsInitialized)
            {
                throw new InvalidOperationException("Data is not initialized.");
            }
        }

        /// <summary>
        /// Gets the initialized DataBuffer of type G from a untyped DataBuffer.
        /// </summary>
        /// <typeparam name="G">The type of the DataBuffer.</typeparam>
        /// <param name="buffer">The untyped DataBuffer.</param>
        /// <returns>The already initialized <see cref="DataBuffer{G}"/>.</returns>
        /// <remarks>
        /// Throws an InvalidOperationException if the buffer is not of the expected type or is not initialized.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the buffer is not of the expected type or is not initialized.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static DataBuffer<G> GetInitializedBuffer<G>(DataBuffer? buffer)
            where G : struct, INumber<G>
        {
            if (buffer is not DataBuffer<G> typeBuffer)
            {
                throw new InvalidOperationException($"Expected buffer of type {typeof(DataBuffer<G>).Name}, but got {buffer?.GetType().Name ?? "null"}.");
            }
            if (!typeBuffer.IsInitialized)
            {
                throw new InvalidOperationException("Buffer is not initialized.");
            }
            return typeBuffer;
        }
    }
}
