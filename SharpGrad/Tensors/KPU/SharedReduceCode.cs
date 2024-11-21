using System;

namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// Shared code for reductions functions.
    /// </summary>
    public enum SharedReduceCode : short // short[0:2]
    {
        /// <summary>
        /// Get the minimum @SafeAccelerator
        /// </summary>
        Min = 0,

        /// <summary>
        /// Get the maximum @SafeAccelerator
        /// </summary>
        Max = 1,

        /// <summary>
        /// Get the variance
        /// </summary>
        Var = 2,

        /// <summary>
        /// Get the standard deviation
        /// </summary>
        Std = 3,

        /// <summary>
        /// Get the sum
        /// </summary>
        Sum = SharedOpCode.Add, // 4

        /// <summary>
        /// Get the product
        /// </summary>
        Prod = SharedOpCode.Mul, // 5

        /// <summary>
        /// Get the mean @SafeAccelerator
        /// </summary>
        Mean = 6,

        //Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine7 = 7,
    }
}