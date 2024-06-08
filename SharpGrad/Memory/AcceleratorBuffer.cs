using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using SharpGrad.Tensors;
using SharpGrad.Tensors.Operators;
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
    internal interface ICreateAcceleratorBuffer<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        abstract static AcceleratorBuffer<T> Create(long length);
        abstract static AcceleratorBuffer<T> Create(T[] data);
        abstract static AcceleratorBuffer<T> Create(AcceleratorBuffer<T> data);
        abstract static AcceleratorBuffer<T> Create(MemoryBuffer1D<T, Stride1D.Dense> data);
    }

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
    public class AcceleratorBuffer<T> : AcceleratorBuffer, IAcceleratorBuffer<T>, IReadOnlyList<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        /// <summary>
        /// Get or set the threshold to force the data to be copied to the <see cref="Accelerator"/>.
        /// </summary>
        public static int CopyToAcceleratorThreshold = 512;

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
                acceleratorData?.Dispose();
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
                                Acc.Synchronize();
                                acceleratorData.Dispose();
                                acceleratorData = null;
                            }
                            break;
                        case BufferLocation.Accelerator:
                            acceleratorData ??= Acc.Allocate1D<T>(Length);
                            if (cpuData is not null)
                            {
                                acceleratorData = Acc.Allocate1D(CPUData);
                                Acc.Synchronize();
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
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Ram"/>.</remarks>
        public T this[int index]
        {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Accelerator"/>.</remarks>
        public T this[Index1D index]
        {
            get => AcceleratorData.View[index];
            set => AcceleratorData.View[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        protected AcceleratorBuffer(long length)
            : base(length) { }


        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        protected AcceleratorBuffer(T[] data)
            : this(data.Length) { cpuData = data; }
        protected AcceleratorBuffer(MemoryBuffer1D<T, Stride1D.Dense> data)
            : this(data.Length) { acceleratorData = data; }

        // Create a new DeviceBuffer with the specified length.
        internal static AcceleratorBuffer<T> Create(long length) => new(length);
        // Create a new DeviceBuffer with the specified data.
        internal static AcceleratorBuffer<T> Create(T[] data) => new(data);
        // Create a new DeviceBuffer with the specified data.
        internal static AcceleratorBuffer<T> Create(AcceleratorBuffer<T> data)
        {
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
        // Create a new DeviceBuffer with the specified data.
        internal static AcceleratorBuffer<T> Create(MemoryBuffer1D<T,Stride1D.Dense> data)
        {
            AcceleratorBuffer<T> buffer = new(data);
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

        public static void ExecInPlace<TOP>(AcceleratorBuffer<T> @this)
            where TOP : IExecutor1<T, T>
        {
            if (@this.Location == BufferLocation.Ram && @this.Length < CopyToAcceleratorThreshold)
                for (int i = 0; i < @this.Length; i++)
                    @this.CPUData[i] = TOP.Exec(@this.CPUData[i]);
            else
                Acc.ExecInPlace<TOP, T>(@this.AcceleratorData);
        }


        public static void ExecInPlace<TOP>(AcceleratorBuffer<T> @this, AcceleratorBuffer<T> other)
            where TOP : IExecutor2<T, T, T>
        {
            if (@this.Length != other.Length)
                throw new ArgumentException($"Expected for same length. {nameof(@this)}.Length = {@this.Length}, {nameof(other)}.Length = {other.Length}");

            if (@this.Location == BufferLocation.Ram && @this.Length < CopyToAcceleratorThreshold)
                for (int i = 0; i < @this.Length; i++)
                    @this.CPUData[i] = TOP.Exec(@this.CPUData[i], other.CPUData[i]);
            else
            {
                @this.Location = BufferLocation.Accelerator;
                other.Location = BufferLocation.Accelerator;
                Acc.Synchronize();
                Acc.ExecInPlace<TOP, T>(@this.AcceleratorData, other.AcceleratorData);
            }
        }
        public void Add(AcceleratorBuffer<T> other) => ExecInPlace<AddOp<T>>(this, other);
        public void Sub(AcceleratorBuffer<T> other) => ExecInPlace<SubOp<T>>(this, other);
        public void Neg() => ExecInPlace<NegOp<T>>(this);
        public void Mul(AcceleratorBuffer<T> other) => ExecInPlace<MulOp<T>>(this, other);
        public void Div(AcceleratorBuffer<T> other) => ExecInPlace<DivOp<T>>(this, other);

        public static AcceleratorBuffer<T> operator +(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
            => Acc.GetAcceleratorBuffer(Acc.Exec<AddOp<T>, T>(left.AcceleratorData, right.AcceleratorData));
        public static AcceleratorBuffer<T> operator -(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
            => Acc.GetAcceleratorBuffer(Acc.Exec<SubOp<T>, T>(left.AcceleratorData, right.AcceleratorData));
        public static AcceleratorBuffer<T> operator -(AcceleratorBuffer<T> left)
            => Acc.GetAcceleratorBuffer(Acc.Exec<NegOp<T>, T>(left.AcceleratorData));
        public static AcceleratorBuffer<T> operator *(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
            => Acc.GetAcceleratorBuffer(Acc.Exec<MulOp<T>, T>(left.AcceleratorData, right.AcceleratorData));
        public static AcceleratorBuffer<T> operator /(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
            => Acc.GetAcceleratorBuffer(Acc.Exec<DivOp<T>, T>(left.AcceleratorData, right.AcceleratorData));


        public static implicit operator T[](AcceleratorBuffer<T> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<T, Stride1D.Dense>(AcceleratorBuffer<T> gpu) => gpu.AcceleratorData;
        public static explicit operator T(AcceleratorBuffer<T> gpu) => gpu.Length == 1 ? gpu.CPUData[0] : throw new InvalidCastException($"Cannot cast a buffer of length {gpu.Length} to a scalar.");
        public static implicit operator AcceleratorBuffer<T>(T cpu) => new([cpu]);
    }

    public class AcceleratorBufferReal<T> : AcceleratorBuffer<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        protected AcceleratorBufferReal(long length) : base(length) { }

        protected AcceleratorBufferReal(T[] data) : base(data) { }

        protected AcceleratorBufferReal(MemoryBuffer1D<T, Stride1D.Dense> data) : base(data) { }

        public void Pow(AcceleratorBuffer<T> right)
        {
            if (right.Length != Length)
                throw new ArgumentException($"Expected length {Length}, got {right.Length}");

            if (Location == BufferLocation.Ram && Length < CopyToAcceleratorThreshold)
            {
                for (int i = 0; i < Length; i++)
                    CPUData[i] = T.Pow(CPUData[i], right.CPUData[i]);
            }
            else
            {
                right.Location = BufferLocation.Accelerator;
                Acc.Synchronize();
                AcceleratorData = Acc.Exec<PowOp<T>, T>(AcceleratorData, right.AcceleratorData);
            }
        }
        public static AcceleratorBuffer<T> Pow(AcceleratorBuffer<T> left, AcceleratorBuffer<T> right)
            => Acc.GetAcceleratorBuffer(Acc.Exec<PowOp<T>, T>(left.AcceleratorData, right.AcceleratorData));


        public void Log()
        {
            if (Location == BufferLocation.Ram && Length < CopyToAcceleratorThreshold)
            {
                for (int i = 0; i < Length; i++)
                    CPUData[i] = T.Log(CPUData[i]);
            }
            else
            {
                AcceleratorData = Acc.Exec<LogOp, T>(AcceleratorData);
            }
        }
        public static AcceleratorBuffer<T> Log(AcceleratorBuffer<T> value)
            => Acc.GetAcceleratorBuffer(Acc.Exec<LogOp<T>, T>(value.AcceleratorData));


        internal new static AcceleratorBufferReal<T> Create(long length) { return new AcceleratorBufferReal<T>(length); }
        internal new static AcceleratorBufferReal<T> Create(T[] data) { return new AcceleratorBufferReal<T>(data); }
        internal new static AcceleratorBufferReal<T> Create(MemoryBuffer1D<T, Stride1D.Dense> data) { return new AcceleratorBufferReal<T>(data); }

    }
}