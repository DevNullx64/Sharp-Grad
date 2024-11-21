using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using SharpGrad.Tensors.Formula;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad.Memory
{

    /// <summary>
    /// A structure that manages data on the RAM and a <see cref="SafeAccelerator"/> (GPU). It free the RAM data when the data is available on the <see cref="SafeAccelerator"/>. And vice versa.
    /// </summary>
    public abstract class AcceleratorBuffer : IUnsafeAcceleratorBuffer
    {
        private bool disposedValue;
        private readonly ReaderWriterLockSlim rwLock = new();

        internal IUnsafeAcceleratorBuffer UnsafeBuffer => this;

        protected void EnterUpgradeableReadLock()
        {
            rwLock.EnterUpgradeableReadLock();
            LastAccess = DateTime.UtcNow.Ticks;
        }
        protected void ExitUpgradeableReadLock() => rwLock.ExitUpgradeableReadLock();
        void IUnsafeAcceleratorBuffer.EnterUpgradeableLock() => EnterUpgradeableReadLock();
        void IUnsafeAcceleratorBuffer.ExitUpgradeableReadLock() => ExitUpgradeableReadLock();
        public bool IsUpgradeableReadLockHeld => rwLock.IsUpgradeableReadLockHeld;

        protected void EnterReadLock()
        {
            rwLock.EnterReadLock();
            LastAccess = DateTime.UtcNow.Ticks;
        }
        protected void ExitReadLock() => rwLock.ExitReadLock();
        void IUnsafeAcceleratorBuffer.EnterReadLock() => EnterReadLock();
        void IUnsafeAcceleratorBuffer.ExitReadLock() => ExitReadLock();
        public bool IsReadLockHeld => rwLock.IsReadLockHeld;

        protected void EnterExclusiveLock()
        {
            rwLock.EnterWriteLock();
            LastAccess = DateTime.UtcNow.Ticks;
        }
        protected void ExitExclusiveLock() => rwLock.ExitWriteLock();
        void IUnsafeAcceleratorBuffer.EnterExclusiveLock() => EnterExclusiveLock();
        void IUnsafeAcceleratorBuffer.ExitExclusiveLock() => ExitExclusiveLock();
        public bool IsExclusiveLockHeld => rwLock.IsWriteLockHeld;

        public bool IsNoLockHeld => !(rwLock.IsReadLockHeld || rwLock.IsWriteLockHeld || rwLock.IsUpgradeableReadLockHeld);

        private AcceleratorExtender accelerator;
        protected AcceleratorExtender UnsafeAccelerator
        {
            get => accelerator;
            set
            {
                if (value != accelerator)
                {
                    if (UnsafeGetLocation() != BufferLocation.Ram)
                        UnsafeSetLocation(BufferLocation.Ram);
                    accelerator = value;
                }
            }
        }
        public AcceleratorExtender SafeAccelerator
        {
            get
            {
                EnterReadLock();
                try { return accelerator; }
                finally { ExitReadLock(); }
            }
            set
            {
                EnterUpgradeableReadLock();
                try
                {
                    if (UnsafeGetLocation() != BufferLocation.Ram)
                    {
                        EnterExclusiveLock();
                        try { UnsafeSetLocation(BufferLocation.Ram); }
                        finally { ExitExclusiveLock(); }
                    }
                }
                finally { ExitUpgradeableReadLock(); }
            }
        }

        AcceleratorExtender IUnsafeAcceleratorBuffer.UnsafeAccelerator
        {
            get => UnsafeAccelerator;
            set => UnsafeAccelerator = value;
        }


        public BufferLocation SafeLocation
        {
            get
            {
                EnterReadLock();
                try { return UnsafeGetLocation(); }
                finally { ExitReadLock(); }
            }
            set
            {
                EnterUpgradeableReadLock();
                try
                {
                    EnterExclusiveLock();
                    try { UnsafeSetLocation(value); }
                    finally { ExitExclusiveLock(); }
                }
                finally { ExitUpgradeableReadLock(); }
            }
        }

        protected abstract BufferLocation UnsafeGetLocation();
        protected abstract void UnsafeSetLocation(BufferLocation value);

        /// <summary>
        /// The location of the data.
        /// </summary>
        BufferLocation IUnsafeAcceleratorBuffer.UnsafeLocation
        {
            get => UnsafeGetLocation();
            set => UnsafeSetLocation(value);
        }


        /// <summary>
        /// The last time the data was accessed on the <see cref="SafeAccelerator"/>.
        /// </summary>
        public long LastAccess { get; protected set; } = DateTime.UtcNow.Ticks;

        /// <summary>
        /// The longLength of the data.
        /// </summary>
        /// <remarks>Prefer SafeAccelerator over other properties that return the longLength of the data.</remarks>
        public long LongLength { get; }

        internal AcceleratorBuffer(AcceleratorExtender accelerator, long longLength)
        {
            this.accelerator = accelerator;
            LongLength = longLength;
        }


        /// <summary>
        /// Dispose the buffer.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    UnsafeBuffer.EnterExclusiveLock();
                    try { UnsafeSetLocation(BufferLocation.Empty); }
                    finally { UnsafeBuffer.ExitExclusiveLock(); }
                    accelerator.Release(this);
                    Dispose();
                }

                // TODO: libérer les ressources non managées (objets non managés) et substituer le finaliseur
                // TODO: affecter aux grands champs une valeur null
                disposedValue = true;
            }
        }

        /// <inheritdoc/>
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
    /// <remarks>If only <paramref name="length"/> is provided, no Buffers will be allocated on the RAM or the <see cref="Accelerator"/>. DataIndices will be allocated and set to zero at the first access.</remarks>
    /// <remarks>
    /// Create a new DeviceBuffer with the specified longLength.
    /// </remarks>
    /// <param name="data">The data to be copied to the RAM.</param>
    /// <remarks><paramref name="data"/> will be copied as reference. But SafeAccelerator link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
    public class AcceleratorBuffer<T>(AcceleratorExtender acceleratorExtender, long length) :
        AcceleratorBuffer(acceleratorExtender, length),
        IExclusiveLockAcceleratorBuffer<T>,
        ILocableAcceleratorBuffer<T>,
        IUpgradableLockAcceleratorBuffer<T>
        where T : unmanaged
    {
        // The data on the RAM.
        private T[]? cpuData = null;
        public bool IsOnRAM => cpuData is not null;

        /// <summary>
        /// Get or set the data on the RAM.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="Accelerator"/>. and the <see cref="Accelerator"/> data will be disposed.</remarks>
        public T[] SafeCPUData
        {
            get
            {
                UnsafeBuffer.EnterUpgradeableLock();
                try
                {
                    if (UnsafeGetLocation() != BufferLocation.Ram)
                    {
                        UnsafeBuffer.EnterExclusiveLock();
                        try { UnsafeSetLocation(BufferLocation.Ram); }
                        finally { UnsafeBuffer.ExitExclusiveLock(); }
                    }
                    LastAccess = DateTime.UtcNow.Ticks;
                    return cpuData!;
                }
                finally { UnsafeBuffer.ExitUpgradeableReadLock(); }
            }

            set
            {
                if (value.Length != LongLength)
                    throw new ArgumentException($"Expected longLength {LongLength}, got {value.Length}");

                UnsafeBuffer.EnterExclusiveLock();
                try
                {
                    UnsafeSetLocation(BufferLocation.Empty);
                    cpuData = value;
                    LastAccess = DateTime.UtcNow.Ticks;
                }
                finally { UnsafeBuffer.ExitExclusiveLock(); }
            }
        }

        protected T[]? UnsafeCPUData
        {
            get => cpuData;
            set
            {
                if (value is not null && value.Length != LongLength)
                    throw new ArgumentException($"Expected longLength {LongLength}, got {value.Length}");

                acceleratorData?.Dispose();
                acceleratorData = null;
                cpuData = value;
                LastAccess = DateTime.UtcNow.Ticks;
            }
        }
        IReadOnlyList<T> IReadOnlyAcceleratorBuffer<T>.SafeCPUData
        {
            get
            {
                if (UnsafeGetLocation() != BufferLocation.Ram)
                {
                    if (IsExclusiveLockHeld)
                        UnsafeSetLocation(BufferLocation.Ram);
                    else if (IsUpgradeableReadLockHeld)
                    {
                        EnterExclusiveLock();
                        try { UnsafeSetLocation(BufferLocation.Ram); }
                        finally { ExitExclusiveLock(); }
                    }
                    else
                        throw new InvalidOperationException($"Buffer is not accessible and cannot be moved to RAM.");
                }
                return UnsafeCPUData!;
            }
        }

        T[]? IExclusiveLockAcceleratorBuffer<T>.SafeCPUData
        {
            get => UnsafeCPUData;
            set => UnsafeCPUData = value;
        }

        /// <summary>
        /// Return a copy of the data.
        /// </summary>
        public T[] GetDataSnapshoot()
        {
            UnsafeBuffer.EnterReadLock();
            try
            {
                switch (UnsafeGetLocation())
                {
                    case BufferLocation.Ram:
                        return (T[])SafeCPUData.Clone();
                    case BufferLocation.Accelerator:
                        T[] data = new T[LongLength];
                        AcceleratorData.CopyToCPU(data);
                        return data;
                    default:
                        return [];
                }
            }
            finally
            {
                UnsafeBuffer.ExitReadLock();
            }
        }

        public bool IsOnAccelerator => acceleratorData is not null;
        // The data on the SafeAccelerator.
        private MemoryBuffer1D<T, Stride1D.Dense>? acceleratorData = null;

        /// <summary>
        /// Get or set the data on the <see cref="Accelerator"/>.
        /// </summary>
        /// <remarks>If data is not available on the <see cref="Accelerator"/>, it will be copied from the RAM. and the RAM data will be disposed.</remarks>
        public ArrayView1D<T, Stride1D.Dense> AcceleratorData
        {
            get
            {
                UnsafeBuffer.EnterUpgradeableLock();
                try
                {
                    if(UnsafeGetLocation() != BufferLocation.Accelerator)
                    {
                        UnsafeBuffer.EnterExclusiveLock();
                        try { UnsafeSetLocation(BufferLocation.Accelerator); }
                        finally { UnsafeBuffer.ExitExclusiveLock(); }
                    }
                    LastAccess = DateTime.UtcNow.Ticks;
                    return AcceleratorData;
                }
                finally { UnsafeBuffer.ExitUpgradeableReadLock(); }
            }
        }


        /// <summary>
        /// Get or set the location of the data.
        /// </summary>
        /// <remarks>Set <see cref="BufferLocation.Empty"/> to free the ressources. If and empty buffer is accessed using the <see cref="AcceleratorData"/>, the data will be allocated but not set to zero.</remarks>
        protected override BufferLocation UnsafeGetLocation()
        {
            if (IsOnRAM)
            {
                return BufferLocation.Ram;
            }
            else if (IsOnAccelerator)
            {
                return BufferLocation.Accelerator;
            }
            else
            {
                return BufferLocation.Empty;
            }
        }

        /// <summary>
        /// Get or set the location of the data.
        /// </summary>
        /// <remarks>Set <see cref="BufferLocation.Empty"/> to free the ressources. If and empty buffer is accessed using the <see cref="AcceleratorData"/>, the data will be allocated but not set to zero.</remarks>
        protected override void UnsafeSetLocation(BufferLocation value)
        {
            if (value == UnsafeGetLocation())
                return;

            switch (value)
            {
                case BufferLocation.Ram:
                    cpuData = new T[LongLength];
                    if (IsOnAccelerator)
                    {
                        acceleratorData!.CopyToCPU(cpuData);
                        acceleratorData!.Dispose();
                        acceleratorData = null;
                    }
                    break;
                case BufferLocation.Accelerator:
                    if (IsOnRAM)
                    {
                        acceleratorData = UnsafeAccelerator.Allocate1D(cpuData!);
                        cpuData = null;
                    }
                    else
                        acceleratorData = UnsafeAccelerator.Allocate1D<T>(LongLength);
                    break;
                case BufferLocation.Empty:
                    cpuData = null;
                    if (IsOnAccelerator)
                    {
                        acceleratorData!.Dispose();
                        acceleratorData = null;
                    }
                    break;
            }
        }

        new public BufferLocation SafeLocation
        {
            get => base.SafeLocation;
            set => base.SafeLocation = value;
        }

        AcceleratorExtender IExclusiveLockAcceleratorBuffer<T>.SafeAccelerator {
            get => UnsafeBuffer.UnsafeAccelerator;
            set => UnsafeBuffer.UnsafeAccelerator = value;
        }

        public ArrayView1D<T, Stride1D.Dense>? SafeAcceleratorData
        {
            get
            {
                EnterReadLock();
                try { return acceleratorData?.View; }
                finally { ExitReadLock(); }
            }
        }
        int IReadOnlyCollection<T>.Count => (int)LongLength;

        T[]? ILocableAcceleratorBuffer<T>.UnsafeCPUData { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public MemoryBuffer1D<T, Stride1D.Dense>? UnsafeAcceleratorData => throw new NotImplementedException();

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Ram"/>.</remarks>
        public T this[int index]
        {
            get => SafeCPUData[index];
            set => SafeCPUData[index] = value;
        }
        T IReadOnlyList<T>.this[int index] => SafeCPUData[index];

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Accelerator"/>.</remarks>
        public T this[Index index]
        {
            get => SafeCPUData[index];
            set => SafeCPUData[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified longLength.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But SafeAccelerator link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        public AcceleratorBuffer(AcceleratorExtender acceleratorExtender, T[] data)
            : this(acceleratorExtender, data.Length)
        {
            ArgumentNullException.ThrowIfNull(data);
            cpuData = data;
        }
        public AcceleratorBuffer(AcceleratorExtender acceleratorExtender, MemoryBuffer1D<T, Stride1D.Dense> data)
            : this(acceleratorExtender, data.Length) {
            if (data.IsDisposed)
                throw new ArgumentException($"The data is disposed.");
            acceleratorData = data; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator<T> GetEnumerator() => SafeCPUData.AsEnumerable().GetEnumerator();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                EnterExclusiveLock();
                try { UnsafeSetLocation(BufferLocation.Empty); }
                finally { ExitExclusiveLock(); }
            }
            base.Dispose(disposing);
        }

        /// <summary>
        /// Check if the data is equal to the specified data.
        /// </summary>
        /// <param name="other">The data to compare with.</param>
        /// <returns>True if the data is equal to the specified data, false otherwise.</returns>
        public bool IsDataReferenceEqual(T[] other)
            => IsOnRAM && ReferenceEquals(SafeCPUData, other);

        /// <summary>
        /// UnsafeFill the data with the specified @SafeAccelerator.
        /// </summary>
        /// <param name="value">The @SafeAccelerator to fill the data with.</param>
        public void UnsafeFill(T value)
        {
            if (SafeLocation == BufferLocation.Ram)
                Array.Fill(SafeCPUData, value);
            else
                UnsafeBuffer.UnsafeAccelerator.Fill(acceleratorData!, value);
        }

        /// <summary>
        /// Set all the data to zero.
        /// </summary>
        public void MemSetToZero()
        {
            if (IsOnRAM)
                Array.Clear(SafeCPUData, 0, SafeCPUData.Length);
            else if (IsOnAccelerator)
                AcceleratorData.MemSetToZero();
        }

        public void CopyTo(AcceleratorBuffer<T> destination)
        {
            if (destination.LongLength != LongLength)
                throw new ArgumentException($"Expected longLength {LongLength}, got {destination.LongLength}");
            if (IsOnRAM)
                destination.SafeCPUData = SafeCPUData;
            else if (IsOnAccelerator)
                destination.AcceleratorData.CopyToCPU(destination.SafeCPUData);
        }

        public void Reset()
        {
            if (IsOnAccelerator)
                AcceleratorData.MemSetToZero();
            else
                Array.Clear(SafeCPUData, 0, SafeCPUData.Length);
        }

        public IReadOnlyAcceleratorBuffer<T> AsReadOnly()
            => new ReadOnlyAcceleratorBuffer<T>(this);

        public IUpgradableLockAcceleratorBuffer<T> AsUpgradableLock()
            => new UpgradableLockAcceleratorBuffer<T>(this);

        public IExclusiveLockAcceleratorBuffer<T> AsWritable()
            => new ExclusiveLockAcceleratorBuffer<T>(this);

        IExclusiveLockAcceleratorBuffer<T> IUpgradableLockAcceleratorBuffer<T>.EnterExclusiveLock()
            => new ExclusiveLockAcceleratorBuffer<T>(this);

        public static implicit operator T[](AcceleratorBuffer<T> gpu) => gpu.SafeCPUData;
        public static implicit operator ArrayView1D<T, Stride1D.Dense>(AcceleratorBuffer<T> gpu) => gpu.AcceleratorData;
        public static explicit operator T(AcceleratorBuffer<T> gpu) => gpu.LongLength == 1
            ? gpu.SafeCPUData[0]
            : throw new InvalidCastException($"Cannot cast a buffer of length {gpu.LongLength} to a scalar.");
    }

    internal readonly struct ReadOnlyAcceleratorBuffer<T> : IReadOnlyAcceleratorBuffer<T>
        where T : unmanaged
    {
        private readonly ILocableAcceleratorBuffer<T> buffer;

        public AcceleratorExtender SafeAccelerator => buffer.UnsafeAccelerator;
        public long LastAccess => buffer.LastAccess;
        public long LongLength => buffer.LongLength;
        public BufferLocation SafeLocation => buffer.UnsafeLocation;

        public IReadOnlyList<T> SafeCPUData => buffer.UnsafeCPUData!;
        public int Count => buffer.Count;

        public T this[int index] => buffer[index];

        public T this[Index index] => buffer[index];

        public ReadOnlyAcceleratorBuffer(ILocableAcceleratorBuffer<T> buffer)
        {
            this.buffer = buffer;
            while (true)
            {
                buffer.EnterExclusiveLock();
                try { buffer.UnsafeLocation = BufferLocation.Ram; }
                finally { buffer.ExitExclusiveLock(); }

                buffer.EnterReadLock();
                try
                {
                    if (buffer.UnsafeLocation == BufferLocation.Ram)
                        break;
                }
                catch
                {
                    buffer.ExitReadLock();
                    throw;
                }
                buffer.ExitReadLock();
            }
        }

        public void Dispose() => buffer.ExitReadLock();

        public IEnumerator<T> GetEnumerator() => buffer.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    internal readonly struct UpgradableLockAcceleratorBuffer<T> : IUpgradableLockAcceleratorBuffer<T>
        where T : unmanaged
    {
        private readonly ILocableAcceleratorBuffer<T> buffer;

        public AcceleratorExtender SafeAccelerator => buffer.UnsafeAccelerator;
        public long LastAccess => buffer.LastAccess;
        public long LongLength => buffer.LongLength;
        public BufferLocation SafeLocation => buffer.UnsafeLocation;

        public IReadOnlyList<T> SafeCPUData => buffer.UnsafeCPUData!;
        public int Count => buffer.Count;

        public T this[int index] => buffer[index];

        public T this[Index index] => buffer[index];

        public UpgradableLockAcceleratorBuffer(ILocableAcceleratorBuffer<T> buffer)
        {
            this.buffer = buffer;
            buffer.EnterUpgradeableLock();
        }

        public IExclusiveLockAcceleratorBuffer<T> EnterExclusiveLock()
            => new ExclusiveLockAcceleratorBuffer<T>(buffer);

        public void Dispose() => buffer.ExitUpgradeableReadLock();

        public IEnumerator<T> GetEnumerator() => buffer.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    internal readonly struct ExclusiveLockAcceleratorBuffer<T> : IExclusiveLockAcceleratorBuffer<T>
        where T : unmanaged
    {
        private readonly ILocableAcceleratorBuffer<T> buffer;

        public long LastAccess => buffer.LastAccess;
        public long LongLength => buffer.LongLength;

        public int Count => buffer.Count;

        public ArrayView1D<T, Stride1D.Dense>? SafeAcceleratorData => buffer.UnsafeAcceleratorData?.View;

        public AcceleratorExtender SafeAccelerator 
        {
            get => buffer.UnsafeAccelerator;
            set => buffer.UnsafeAccelerator = value;
        }

        public BufferLocation SafeLocation
        {
            get => buffer.UnsafeLocation;
            set => buffer.UnsafeLocation = value;
        }

        public T[]? SafeCPUData
        {
            get => buffer.UnsafeCPUData;
            set => buffer.UnsafeCPUData = value;
        }

        IReadOnlyList<T> IReadOnlyAcceleratorBuffer<T>.SafeCPUData
        {
            get
            {
                buffer.UnsafeLocation = BufferLocation.Ram;
                return buffer.UnsafeCPUData!;
            }
        }

        public T this[int index] => buffer[index];

        public T this[Index index] => buffer[index];

        public ExclusiveLockAcceleratorBuffer(ILocableAcceleratorBuffer<T> buffer)
        {
            this.buffer = buffer;
            buffer.EnterExclusiveLock();
        }

        public void Dispose() => buffer.ExitExclusiveLock();

        public IEnumerator<T> GetEnumerator() => buffer.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}