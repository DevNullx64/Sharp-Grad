using System;

namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// Shared code for reductions functions.
    /// </summary>
    public enum SharedReduceCode : short // short[0:2]
    {
        /// <summary>
        /// Get the minimum @this
        /// </summary>
        Min = 0,

        /// <summary>
        /// Get the maximum @this
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
        /// Get the mean @this
        /// </summary>
        Mean = 6,

        //Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine7 = 7,
    }
}