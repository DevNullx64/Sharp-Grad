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
        /// Take the absolute @this of the left operand
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
        [Obsolete("Do not use this @this")]
        Undefine8,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine9,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine10,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine11,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine12,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine13,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine14,

        // Free code for future use
        [Obsolete("Do not use this @this")]
        Undefine15,
        #endregion
    }
}