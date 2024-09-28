using System;

namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// Shared functions code.
    /// </summary>
    /// <remarks>It is a bit field short[0:3] where only short[0:2] is defined at the moment.</remarks>
    public enum SharedFuncCode : short // short[0:4]
    {
        /// <summary>
        /// Negate the left operand
        /// </summary>
        Neg = 0,

        /// <summary>
        /// Take the natural logarithm of the left operand
        /// </summary>
        Log,

        /// <summary>
        /// Take the exponential of the left operand
        /// </summary>
        Exp,

        /// <summary>
        /// Take the square root of the left operand
        /// </summary>
        Sqrt,

        /// <summary>
        /// Take the absolute value of the left operand
        /// </summary>
        Abs,

        /// <summary>
        /// Take the sine of the left operand
        /// </summary>
        Sin,

        /// <summary>
        /// Take the cosine of the left operand
        /// </summary>
        Cos,

        /// <summary>
        /// Take the tangent of the left operand
        /// </summary>
        Tan,

        #region Free codes for future use
        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine8,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine9,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine10,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine11,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine12,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine13,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine14,

        // Free code for future use
        [Obsolete("Do not use this value")]
        Undefine15,
        #endregion
    }
}