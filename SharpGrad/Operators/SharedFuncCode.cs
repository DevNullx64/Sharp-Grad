using System;

namespace SharpGrad.Operators
{
    /// <summary>
    /// Shared functions code.
    /// </summary>
    /// <remarks>It is a bit field short[0:3] where only short[0:2] is defined at the moment.</remarks>
    public enum SharedFuncCode : short // short[0:4]
    {
        /// <summary>
        /// Negate the operand
        /// </summary>
        Neg = 0,

        /// <summary>
        /// Take the natural logarithm of the operand
        /// </summary>
        Log,

        /// <summary>
        /// Take the exponential of the operand
        /// </summary>
        Exp,

        /// <summary>
        /// Take the square root of the operand
        /// </summary>
        Sqrt,

        /// <summary>
        /// Take the absolute @SafeAccelerator of the operand
        /// </summary>
        Abs,

        /// <summary>
        /// Take the sine of the operand
        /// </summary>
        Sin,

        /// <summary>
        /// Take the cosine of the operand
        /// </summary>
        Cos,

        /// <summary>
        /// Take the tangent of the operand
        /// </summary>
        Tan,

        /// <summary>
        /// Take the hyperbolic tangent of the operand
        /// </summary>
        Tanh,

        #region Free codes for future use
        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine9,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine10,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine11,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine12,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine13,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine14,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine15,
        #endregion
    }
}