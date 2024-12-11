using System;

namespace SharpGrad.Operators
{
    /// <summary>
    /// Shared operators code.
    /// </summary>
    /// <remarks>It is a bit field short[0:2].</remarks>
    public enum SharedOpCode : byte // short[0:2]
    {
        /// <summary>
        /// IsCommutative is set for commutative <see cref="SharedOpCode"/>
        /// </summary>
        IsCommutative = 0x0004,

        /// <summary>
        /// Subtract the right operand from the left operand
        /// </summary>
        Sub = 0,

        /// <summary>
        /// Add the left operand to the right operand
        /// </summary>
        Add = Sub | IsCommutative,

        /// <summary>
        /// Divide the left operand by the right operand
        /// </summary>
        Div = 1,

        /// <summary>
        /// Multiply the left operand by the right operand
        /// </summary>
        Mul = Div | IsCommutative,

        /// <summary>
        /// Raise the left operand to the power of the right operand
        /// </summary>
        Pow = 2,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine6 = Pow | IsCommutative,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine3 = 3,

        // Free code for future use
        [Obsolete("Do not use SafeAccelerator @SafeAccelerator")]
        Undefine7 = Undefine3 | IsCommutative,
    }
}