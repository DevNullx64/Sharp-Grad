namespace SharpGrad.Memory
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
        /// Data is available on the <see cref="ILGPU.Runtime.Accelerator"/>.
        /// </summary>
        Accelerator
    }
}