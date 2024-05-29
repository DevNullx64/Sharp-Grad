namespace SharpGrad.Tensors
{
    /// <summary>
    /// The location of the data.
    /// </summary>
    public enum BufferLocation
    {
        /// <summary>
        /// No data is available.
        /// </summary>
        Empty,
        /// <summary>
        /// Data is available on the RAM.
        /// </summary>
        Ram,
        /// <summary>
        /// Data is available on the shared memory.
        /// </summary>
        SharedMemory,
        /// <summary>
        /// Data is available on the <see cref="ILGPU.Runtime.Accelerator"/>.
        /// </summary>
        Accelerator
    }
}