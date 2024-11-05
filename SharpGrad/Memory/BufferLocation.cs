using System;

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
        /// DataIndices is available on the <see cref="ILGPU.Runtime.Accelerator"/>.
        /// </summary>
        Accelerator,
        /// <summary>
        /// DataIndices is available on the RAM.
        /// </summary>
        Ram,
        /// <summary>
        /// DataIndices is available on the local storage.
        /// </summary>
        //[Obsolete("/!\\ Not implemented /!\\")]
        //LocalStorage,
        /// <summary>
        /// DataIndices is available on the distributed hash table.
        /// </summary>
        //[Obsolete("/!\\ Not implemented /!\\")]
        //DHT,
    }
}