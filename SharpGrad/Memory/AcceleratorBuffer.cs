using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using SharpGrad.Tensors;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad.Memory
{
    public abstract class AcceleratorBuffer(long length) : IAcceleratorBuffer
    {
        /// <summary>
        /// The last time the data was accessed on the <see cref="Accelerator"/>.
        /// </summary>
        public long LastAccess { get; protected set; } = DateTime.UtcNow.Ticks;

        /// <summary>
        /// The length of the data.
        /// </summary>
        /// <remarks>Prefer this over other properties that return the length of the data.</remarks>
        public long Length { get; } = length;

        private bool disposedValue;

        /// <summary>
        /// The location of the data.
        /// </summary>
        public abstract BufferLocation Location { get; set; }

        /// <summary>
        /// Is the buffer empty (ressource free).
        /// </summary>
        public bool IsEmpty { get => Location == BufferLocation.Empty; }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    Acc.Dispose(this);
                }

                // TODO: libérer les ressources non managées (objets non managés) et substituer le finaliseur
                // TODO: affecter aux grands champs une valeur null
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// A structure that manages data on the RAM and a <see cref="Accelerator"/> (GPU). It free the RAM data when the data is available on the <see cref="Accelerator"/>. And vice versa.
    /// </summary>
    /// <typeparam name="T">The type of the data</typeparam>
    /// <remarks>If only <paramref name="length"/> is provided, no memory will be allocated on the RAM or the <see cref="Accelerator"/>. Data will be allocated and set to zero at the first access.</remarks>
    public class AcceleratorBuffer<T> : AcceleratorBuffer, IAcceleratorBuffer<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        // The data on the RAM.
        private T[]? cpuData = null;

        /// <summary>
        /// Get or set the data on the RAM.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="Accelerator"/>. and the <see cref="Accelerator"/> data will be disposed.</remarks>
        public T[] CPUData
        {
            get
            {
                Location = BufferLocation.Ram;
                return cpuData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                cpuData = value;
                acceleratorData?.Dispose();
                acceleratorData = null;
            }
        }

        // The data on the Accelerator.
        private MemoryBuffer1D<T, Stride1D.Dense>? acceleratorData = null;
        /// <summary>
        /// Get or set the data on the <see cref="Accelerator"/>.
        /// </summary>
        /// <remarks>If data is not available on the <see cref="Accelerator"/>, it will be copied from the RAM. and the RAM data will be disposed.</remarks>
        public MemoryBuffer1D<T, Stride1D.Dense> AcceleratorData
        {
            get
            {
                Location = BufferLocation.Accelerator;
                LastAccess = DateTime.UtcNow.Ticks;
                return acceleratorData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                acceleratorData = value;
                cpuData = null;
            }
        }

        // Implementing and hide the IReadOnlyList<TType> interface.
        int IReadOnlyCollection<T>.Count => (int)Length;

        // Implementing and hide the IReadOnlyList<TType> interface.
        private bool IsOnRAM => cpuData is not null;

        // Implementing and hide the IReadOnlyList<TType> interface.
        private bool IsOnAccelerator => acceleratorData is not null;

        /// <summary>
        /// Get or set the location of the data.
        /// </summary>
        /// <remarks>Set <see cref="BufferLocation.Empty"/> to free the ressources.</remarks>
        public override BufferLocation Location {
            get {
                return IsOnRAM
                    ? BufferLocation.Ram
                    : IsOnAccelerator
                        ? BufferLocation.Accelerator
                        : BufferLocation.Empty;
            }
            set
            {
                if (value != Location)
                {
                    switch (value)
                    {
                        case BufferLocation.Ram:
                            cpuData ??= new T[Length];
                            if (acceleratorData is not null)
                            {
                                acceleratorData.CopyToCPU(cpuData);
                                acceleratorData.Dispose();
                                acceleratorData = null;
                            }
                            break;
                        case BufferLocation.Accelerator:
                            acceleratorData ??= Acc.Allocate1D<T>(Length);
                            if (cpuData is not null)
                            {
                                acceleratorData = Acc.Allocate1D(CPUData);
                                cpuData = null;
                            }
                            else
                                acceleratorData.MemSetToZero();
                            break;
                        case BufferLocation.Empty:
                            cpuData = null;
                            acceleratorData?.Dispose();
                            acceleratorData = null;
                            break;
                    }
                }
            }
        }

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing the data will set the <see cref="Location"/> to <see cref="BufferLocation.Ram"/>.</remarks>
        public T this[int index]
        {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        protected AcceleratorBuffer(long length)
            : base(length) { }

        // Create a new DeviceBuffer with the specified length.
        internal static AcceleratorBuffer<T> Create(long length) => new(length);

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        protected AcceleratorBuffer(T[] data)
            : this(data.Length) { cpuData = data; }

        // Create a new DeviceBuffer with the specified length.
        internal static AcceleratorBuffer<T> Create(T[] data) => new(data);
        internal static AcceleratorBuffer<T> Create(AcceleratorBuffer<T> data) {
            AcceleratorBuffer<T> buffer = new(data.Length)
            {
                Location = data.Location
            };
            if (buffer.IsOnRAM)
                Array.Copy(data.CPUData, buffer.CPUData, data.CPUData.Length);
            else if (buffer.IsOnAccelerator)
                buffer.AcceleratorData.CopyFrom(data.AcceleratorData);
            return buffer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator<T> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                cpuData = null;
                acceleratorData?.Dispose();
                acceleratorData = null;
                Acc.Dispose(this);
            }
        }

        /// <summary>
        /// Fill the data with the specified value.
        /// </summary>
        /// <param name="value">The value to fill the data with.</param>
        public void Fill(T value)
        {
            if (Location == BufferLocation.Ram)
                Array.Fill(CPUData, value);
            else
                Acc.Fill(AcceleratorData, value);

        }

        /// <summary>
        /// Set all the data to zero.
        /// </summary>
        public void MemSetToZero()
        {
            if (IsOnRAM)
                Array.Clear(CPUData, 0, CPUData.Length);
            else if (IsOnAccelerator)
                AcceleratorData.MemSetToZero();
        }

        public static AcceleratorBuffer<T> operator +(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
        {
            var result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(AddOp<T>.Exec, left.AcceleratorData, right.AcceleratorData, result.AcceleratorData);
            return result;
        }

        public static AcceleratorBuffer<T> operator -(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
        {
            var result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(SubOp<T>.Exec, left.AcceleratorData, right.AcceleratorData, result.AcceleratorData);
            return result;
        }

        public static AcceleratorBuffer<T> operator -(AcceleratorBuffer<T> left)
        {
            var result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(NegOp<T>.Exec, left.AcceleratorData, result.AcceleratorData);
            return result;
        }

        public static AcceleratorBuffer<T> operator *(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
        {
            var result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(MulOp<T>.Exec, left.AcceleratorData, right.AcceleratorData, result.AcceleratorData);
            return result;
        }

        public static AcceleratorBuffer<T> operator /(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
        {
            var result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(DivOp<T>.Exec, left.AcceleratorData, right.AcceleratorData, result.AcceleratorData);
            return result;
        }

        public static implicit operator T[](AcceleratorBuffer<T> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<T, Stride1D.Dense>(AcceleratorBuffer<T> gpu) => gpu.AcceleratorData;
    }
}